#!/usr/bin/env python3
"""
02_train_base_chat_fast.py

FastGPT base training on a chat corpus (pretokenized .npy).
Windows-safe, single GPU, no DataLoader multiprocessing.

Key upgrades vs codex version:
- Safer random-window sampling (avoid starting right after EOS / too close to end)
- Better sampling (top-p + repetition penalty) so samples look less insane early
- ETA + tok/s + save best.pt on improved loss
- Saves/loads GradScaler state for clean resume
- No torch.compile / inductor
"""

import os
import math
import time
import csv
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm
from torch.optim.lr_scheduler import LambdaLR

# ──────────────────────────────
# CONFIG (Model)
# ──────────────────────────────
BLOCK_SIZE      = 768
EMBED_DIM       = 896
NUM_HEADS       = 14
NUM_LAYERS      = 16
DROPOUT         = 0.1

# ──────────────────────────────
# CONFIG (Training)
# ──────────────────────────────
BATCH_SIZE      = 16
GRAD_ACCUM      = 8

LEARNING_RATE   = 8e-5
WARMUP_STEPS    = 1500

MAX_STEPS       = 20_000

PRINT_EVERY     = 50
SAMPLE_EVERY    = 500
SAVE_EVERY      = 5000

USE_AMP_FP16    = True   # fp16 AMP for RTX (usually fastest)
GRAD_CLIP       = 1.0

# Early stop (optional)
MIN_STEPS_BEFORE_EARLY_STOP = 5000
EARLY_STOP_LOSS = 1.40
EARLY_STOP_PATIENCE = 8  # number of PRINT_EVERY windows

# ──────────────────────────────
# PATHS
# ──────────────────────────────
TOKENIZER_PATH  = "llm_tokenizer.model"
TOKENS_PATH     = "data/chat_corpus_v1_tokens.npy"

CHECKPOINT_DIR  = "checkpoints_chat_v1"
CSV_LOG_PATH    = "training_chat_v1.csv"

# ──────────────────────────────
# Sampling prompts
# ──────────────────────────────
SAMPLE_PROMPTS = [
    "User: Hi\nAssistant:",
    "User: How are you?\nAssistant:",
    "User: What's up?\nAssistant:",
    "User: I feel stressed. Any quick tips?\nAssistant:",
    "User: What's a good way to relax after work?\nAssistant:",
    "User: What is 2+2?\nAssistant:",
    "User: What is 20/5?\nAssistant:",
    "User: What is a transformer model?\nAssistant:",
]

def clean_prompt(text: str) -> str:
    text = text.replace("<unk>", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()

# ──────────────────────────────
# DATASET (Pretokenized random windows)
# ──────────────────────────────
class RandomWindowTokens(IterableDataset):
    """
    Random windows from token memmap.

    Improvements:
    - Avoid starting too close to end (obvious)
    - Avoid starts that land immediately after EOS (often creates weird "mid-example" windows)
    - Tiny rejection sampling for better window quality
    """
    def __init__(self, tokens_path: str, block_size: int, eos_id: int, seed: int = 1234):
        super().__init__()
        self.tokens_path = tokens_path
        self.block_size = block_size
        self.eos_id = int(eos_id) if eos_id is not None else -1
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + int(time.time()))
        tokens = np.load(self.tokens_path, mmap_mode="r")
        n = len(tokens)
        if n < self.block_size + 2:
            raise RuntimeError(f"Token file too small: {n} tokens")

        max_start = n - (self.block_size + 1)

        while True:
            # rejection sampling to avoid pathological starts
            for _ in range(8):
                start = int(rng.integers(0, max_start))
                if self.eos_id >= 0:
                    # avoid starting immediately after eos (often splits conversations badly)
                    if start > 0 and int(tokens[start - 1]) == self.eos_id:
                        continue
                    # also avoid windows that are "almost all eos" (rare but can happen)
                    # quick check on a few positions
                    if int(tokens[start]) == self.eos_id:
                        continue
                break

            seq = tokens[start : start + self.block_size + 1]
            # NOTE: converting small slices to torch is cheap; memmap stays on disk/OS cache
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            yield x, y, start

# ──────────────────────────────
# MODEL
# ──────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.proj_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class FastGPT(nn.Module):
    def __init__(self, vocab, dim, heads, layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(block_size, dim)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        print(f"🔧 Model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T = x.shape
        if T > self.block_size:
            x = x[:, -self.block_size:]
            T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok(x) + self.pos(pos))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))

# ──────────────────────────────
# LOGGING / CHECKPOINTS
# ──────────────────────────────
def init_csv(csv_path: str, resume: bool):
    if resume and os.path.exists(csv_path):
        print(f"📊 Resuming CSV log: {csv_path}")
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","step","loss","ppl","lr","step_s","tok_s","eta","vram_used_gb","vram_max_gb"])
    print(f"📊 Created CSV log: {csv_path}")

def log_csv(csv_path: str, step: int, loss: float, ppl: float, lr: float,
            step_s: float, tok_s: float, eta_str: str,
            vram_used=None, vram_max=None):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.now().isoformat(),
            step,
            f"{loss:.6f}",
            f"{ppl:.4f}",
            f"{lr:.8f}",
            f"{step_s:.4f}",
            f"{tok_s:.1f}",
            eta_str,
            f"{vram_used:.4f}" if vram_used is not None else "",
            f"{vram_max:.4f}" if vram_max is not None else "",
        ])

def latest_ckpt(dir_path: str):
    if not os.path.exists(dir_path):
        return None
    cands = []
    for fn in os.listdir(dir_path):
        if fn.endswith(".pt") and (fn.startswith("step") or fn.startswith("interrupted_step")):
            m = re.findall(r"\d+", fn)
            if m:
                cands.append((int(m[0]), fn))
    if not cands:
        autosave = os.path.join(dir_path, "autosave.pt")
        return autosave if os.path.exists(autosave) else None
    cands.sort()
    return os.path.join(dir_path, cands[-1][1])

def save_ckpt(path: str, step: int, model, opt, sched, scaler, best_loss: float):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "best_loss": float(best_loss),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)

# ──────────────────────────────
# SAMPLING (top-p + repetition penalty)
# ──────────────────────────────
@torch.no_grad()
def generate(
    model,
    sp,
    device,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.55,
    top_p: float = 0.90,
    repetition_penalty: float = 1.05,
    rep_window: int = 64,
):
    """
    More stable chat sampling for early training:
    - lower temperature (default 0.55)
    - nucleus sampling (top_p)
    - light repetition penalty on last rep_window tokens
    """
    model.eval()

    ids = sp.encode(clean_prompt(prompt), out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    eos = sp.eos_id()
    generated = ids.copy()

    for _ in range(max_new_tokens):
        # crop to block size if needed
        if x.size(1) > model.block_size:
            x_in = x[:, -model.block_size:]
        else:
            x_in = x

        logits = model(x_in)[:, -1, :]  # (1, vocab)

        # repetition penalty (last rep_window tokens)
        if repetition_penalty is not None and repetition_penalty > 1.0:
            recent = generated[-rep_window:]
            if recent:
                for tok in set(recent):
                    logits[0, tok] /= repetition_penalty

        # temperature
        if temperature is not None and temperature > 0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)

        # nucleus (top-p)
        if top_p is not None and 0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            # mask tokens beyond top_p
            sorted_probs[cum > top_p] = 0.0
            # renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(sorted_probs, 1)
            next_tok = sorted_idx.gather(-1, next_id).item()
        else:
            next_tok = torch.multinomial(probs, 1).item()

        generated.append(next_tok)
        x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)

        if eos is not None and next_tok == eos:
            break

    model.train()
    return sp.decode(generated)

# ──────────────────────────────
# TRAIN
# ──────────────────────────────
def format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if device == "cuda":
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        # TF32 helps speed on NVIDIA (safe for training)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    vocab = sp.get_piece_size()
    eos_id = sp.eos_id()
    print(f"📚 Vocab size: {vocab:,} | EOS id: {eos_id}")

    # Token file sanity
    tokens_np = np.load(TOKENS_PATH, mmap_mode="r")
    total_tokens = int(len(tokens_np))
    print(f"🧾 Token file: {TOKENS_PATH} | tokens={total_tokens:,}")
    del tokens_np

    # Dataset / loader (single-worker for Windows)
    ds = RandomWindowTokens(TOKENS_PATH, BLOCK_SIZE, eos_id=eos_id)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=(device == "cuda"))

    # Model
    model = FastGPT(vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min((s + 1) / WARMUP_STEPS, 1.0))

    scaler = None
    if device == "cuda" and USE_AMP_FP16:
        scaler = torch.cuda.amp.GradScaler()

    # Resume
    ckpt_path = latest_ckpt(CHECKPOINT_DIR)
    step = 0
    best_loss = float("inf")
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sched"])
        step = int(ckpt.get("step", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"🔁 Resumed: {ckpt_path} (step {step})")

        # force LR to config (optional but often desired)
        for g in optimizer.param_groups:
            g["lr"] = LEARNING_RATE

    init_csv(CSV_LOG_PATH, resume=(ckpt_path is not None))

    model.train()
    micro = 0
    running_loss = 0.0
    early_stop_streak = 0

    it = iter(dl)
    pbar = tqdm(total=MAX_STEPS, initial=step, desc="Training")

    # timers for ETA / throughput
    start_t = time.time()
    last_print_t = start_t

    # tokens per optimizer step = (B * T) * GRAD_ACCUM
    tokens_per_opt_step = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM

    try:
        while step < MAX_STEPS:
            x, y, _pos = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            micro += 1
            if micro % GRAD_ACCUM == 1:
                optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1)) / GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1)) / GRAD_ACCUM
                loss.backward()

            if micro % GRAD_ACCUM == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

                scheduler.step()
                step += 1
                running_loss += loss.item() * GRAD_ACCUM

                # logging
                if step % PRINT_EVERY == 0:
                    avg = running_loss / PRINT_EVERY
                    ppl = math.exp(min(avg, 20))
                    lr = scheduler.get_last_lr()[0]

                    now = time.time()
                    dt = max(1e-6, now - last_print_t)
                    step_s = PRINT_EVERY / dt
                    last_print_t = now

                    tok_s = step_s * tokens_per_opt_step

                    steps_left = MAX_STEPS - step
                    eta = steps_left / max(1e-9, step_s)
                    eta_str = format_eta(eta)

                    vram_used = vram_max = None
                    if device == "cuda":
                        torch.cuda.synchronize()
                        vram_used = torch.cuda.memory_reserved() / 1024**3
                        vram_max = torch.cuda.max_memory_reserved() / 1024**3

                    postfix = {
                        "loss": f"{avg:.4f}",
                        "ppl": f"{ppl:.2f}",
                        "lr": f"{lr:.2e}",
                        "step/s": f"{step_s:.2f}",
                        "tok/s": f"{tok_s:.0f}",
                        "ETA": eta_str,
                    }
                    if vram_used is not None:
                        postfix["VRAM"] = f"{vram_used:.2f}/{vram_max:.2f}GB"

                    pbar.set_postfix(postfix)
                    log_csv(CSV_LOG_PATH, step, avg, ppl, lr, step_s, tok_s, eta_str, vram_used, vram_max)

                    # save best model (by avg window loss)
                    if avg < best_loss:
                        best_loss = avg
                        best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
                        save_ckpt(best_path, step, model, optimizer, scheduler, scaler, best_loss)
                        print(f"\n🏆 New best: loss {best_loss:.4f} @ step {step} → {best_path}")

                    running_loss = 0.0

                    # early stop logic
                    if step >= MIN_STEPS_BEFORE_EARLY_STOP:
                        if avg <= EARLY_STOP_LOSS:
                            early_stop_streak += 1
                        else:
                            early_stop_streak = 0
                        if early_stop_streak >= EARLY_STOP_PATIENCE:
                            print(
                                f"\n🛑 Early stop: loss <= {EARLY_STOP_LOSS} "
                                f"for {EARLY_STOP_PATIENCE} consecutive windows."
                            )
                            break

                # sampling
                if step % SAMPLE_EVERY == 0:
                    prompt = SAMPLE_PROMPTS[(step // SAMPLE_EVERY) % len(SAMPLE_PROMPTS)]
                    out = generate(
                        model, sp, device, prompt,
                        max_new_tokens=120,
                        temperature=0.55,
                        top_p=0.90,
                        repetition_penalty=1.05,
                    )
                    print("\n💬 SAMPLE\n" + out + "\n")

                    autosave = os.path.join(CHECKPOINT_DIR, "autosave.pt")
                    save_ckpt(autosave, step, model, optimizer, scheduler, scaler, best_loss)
                    print(f"💾 Autosaved: {autosave}")

                # periodic checkpoint
                if step % SAVE_EVERY == 0:
                    path = os.path.join(CHECKPOINT_DIR, f"step{step}.pt")
                    save_ckpt(path, step, model, optimizer, scheduler, scaler, best_loss)
                    print(f"💾 Saved checkpoint: {path}")

                pbar.update(1)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted. Saving emergency checkpoint...")
        path = os.path.join(CHECKPOINT_DIR, f"interrupted_step{step}.pt")
        save_ckpt(path, step, model, optimizer, scheduler, scaler, best_loss)
        print(f"💾 Saved: {path}")
        pbar.close()
        return

    final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
    save_ckpt(final_path, step, model, optimizer, scheduler, scaler, best_loss)
    print(f"\n✅ Done. Final checkpoint: {final_path}")
    pbar.close()

if __name__ == "__main__":
    train()
