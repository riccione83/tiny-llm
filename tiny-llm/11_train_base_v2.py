#!/usr/bin/env python3
"""
06_train_base_v2.py

Modernized GPT-style base training (RMSNorm + SwiGLU + RoPE).
Separate from v1 to keep checkpoint compatibility.
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
import sentencepiece as spm
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# -----------------------------
# CONFIG (Model)
# -----------------------------
BLOCK_SIZE      = 768
EMBED_DIM       = 896
NUM_HEADS       = 14
NUM_LAYERS      = 16
DROPOUT         = 0.1

# -----------------------------
# CONFIG (Training)
# -----------------------------
BATCH_SIZE      = 16
GRAD_ACCUM      = 8

LEARNING_RATE   = 8e-5
WARMUP_STEPS    = 1500
MAX_STEPS       = 20_000

PRINT_EVERY     = 50
SAMPLE_EVERY    = 500
SAVE_EVERY      = 5000

USE_MIXED_PRECISION = True
GRAD_CLIP = 1.0

# Early stopping
MIN_STEPS_BEFORE_EARLY_STOP = 5000
EARLY_STOP_LOSS = 1.40
EARLY_STOP_PATIENCE = 8

# -----------------------------
# PATHS
# -----------------------------
TOKENIZER_PATH  = "llm_tokenizer.model"
TOKENS_PATH     = "data/chat_corpus_v1_tokens.npy"

CHECKPOINT_DIR  = "checkpoints_chat_v2"
CSV_LOG_PATH    = "training_chat_v2.csv"

# -----------------------------
# Sampling
# -----------------------------
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

# -----------------------------
# Dataset
# -----------------------------
class RandomWindowTokens(IterableDataset):
    def __init__(self, tokens_path: str, block_size: int, seed: int = 1234):
        super().__init__()
        self.tokens_path = tokens_path
        self.block_size = block_size
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + int(time.time()))
        tokens = np.load(self.tokens_path, mmap_mode="r")
        n = len(tokens)
        if n < self.block_size + 2:
            raise RuntimeError(f"Token file too small: {n} tokens")

        while True:
            start = int(rng.integers(0, n - (self.block_size + 1)))
            seq = tokens[start : start + self.block_size + 1]
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            yield x, y, start

# -----------------------------
# Model (RMSNorm + SwiGLU + RoPE)
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class RoPECache(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len, device):
        return (
            self.cos[..., :seq_len, :].to(device),
            self.sin[..., :seq_len, :].to(device),
        )

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, block_size):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.proj_drop = nn.Dropout(DROPOUT)
        self.rope = RoPECache(self.head_dim, max_seq_len=block_size)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        q, k = apply_rope(q, k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(y))

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.w2 = nn.Linear(dim, 4 * dim, bias=False)
        self.w3 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, dim, heads, block_size):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, block_size)
        self.ln2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class FastGPTv2(nn.Module):
    def __init__(self, vocab, dim, heads, layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, dim)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(dim, heads, block_size) for _ in range(layers)])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        if x.shape[1] > self.block_size:
            x = x[:, -self.block_size:]
        B, T = x.shape
        h = self.drop(self.tok(x))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))

# -----------------------------
# Checkpointing / logging
# -----------------------------
def init_csv(csv_path: str, resume: bool):
    if resume and os.path.exists(csv_path):
        print(f"Resuming CSV log: {csv_path}")
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","step","loss","ppl","lr","vram_used_gb","vram_max_gb"])
    print(f"Created CSV log: {csv_path}")

def log_csv(csv_path: str, step: int, loss: float, ppl: float, lr: float, vram_used=None, vram_max=None):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.now().isoformat(),
            step,
            f"{loss:.6f}",
            f"{ppl:.4f}",
            f"{lr:.8f}",
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

# -----------------------------
# Sampling
# -----------------------------
@torch.no_grad()
def generate(model, sp, device, prompt: str, max_new_tokens=128, temperature=0.7):
    model.eval()
    ids = sp.encode(clean_prompt(prompt), out_type=int)
    x = torch.tensor([ids], device=device)
    eos = sp.eos_id()
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if eos is not None and nxt == eos:
            break
    model.train()
    return sp.decode(x[0].tolist())

# -----------------------------
# Train
# -----------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    vocab = sp.get_piece_size()
    print(f"Vocab size: {vocab:,} | EOS id: {sp.eos_id()}")

    tokens_np = np.load(TOKENS_PATH, mmap_mode="r")
    total_tokens = len(tokens_np)
    print(f"Token file: {TOKENS_PATH} | tokens={total_tokens:,}")
    del tokens_np

    ds = RandomWindowTokens(TOKENS_PATH, BLOCK_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    model = FastGPTv2(vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min((s + 1) / WARMUP_STEPS, 1.0))

    scaler = torch.amp.GradScaler("cuda") if (USE_MIXED_PRECISION and device == "cuda") else None

    ckpt_path = latest_ckpt(CHECKPOINT_DIR)
    step = 0
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sched"])
        step = int(ckpt["step"])
        print(f"Resumed: {ckpt_path} (step {step})")

    init_csv(CSV_LOG_PATH, resume=(ckpt_path is not None))

    model.train()
    micro = 0
    running_loss = 0.0
    last_print_t = time.time()
    early_stop_streak = 0

    it = iter(dl)
    pbar = tqdm(total=MAX_STEPS, initial=step, desc="Training")

    try:
        while step < MAX_STEPS:
            x, y, _pos = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            micro += 1
            if micro % GRAD_ACCUM == 1:
                optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
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

                if step % PRINT_EVERY == 0:
                    avg = running_loss / PRINT_EVERY
                    ppl = math.exp(min(avg, 20))
                    lr = scheduler.get_last_lr()[0]

                    now = time.time()
                    steps_per_sec = PRINT_EVERY / max(1e-6, now - last_print_t)
                    last_print_t = now

                    vram_used = vram_max = None
                    if device == "cuda":
                        torch.cuda.synchronize()
                        vram_used = torch.cuda.memory_reserved() / 1024**3
                        vram_max = torch.cuda.max_memory_reserved() / 1024**3

                    postfix = {"loss": f"{avg:.4f}", "ppl": f"{ppl:.2f}", "lr": f"{lr:.2e}", "step/s": f"{steps_per_sec:.2f}"}
                    if vram_used is not None:
                        postfix["VRAM"] = f"{vram_used:.2f}/{vram_max:.2f}GB"
                    pbar.set_postfix(postfix)

                    log_csv(CSV_LOG_PATH, step, avg, ppl, lr, vram_used, vram_max)
                    running_loss = 0.0

                    if step >= MIN_STEPS_BEFORE_EARLY_STOP:
                        if avg <= EARLY_STOP_LOSS:
                            early_stop_streak += 1
                        else:
                            early_stop_streak = 0
                        if early_stop_streak >= EARLY_STOP_PATIENCE:
                            print(
                                f"Early stop: loss <= {EARLY_STOP_LOSS} for "
                                f"{EARLY_STOP_PATIENCE} consecutive windows."
                            )
                            break

                if step % SAMPLE_EVERY == 0:
                    prompt = SAMPLE_PROMPTS[(step // SAMPLE_EVERY) % len(SAMPLE_PROMPTS)]
                    out = generate(model, sp, device, prompt, max_new_tokens=120, temperature=0.7)
                    print("\nSAMPLE\n" + out + "\n")

                    autosave = os.path.join(CHECKPOINT_DIR, "autosave.pt")
                    torch.save({"step": step, "model": model.state_dict(),
                                "opt": optimizer.state_dict(), "sched": scheduler.state_dict()}, autosave)
                    print(f"Autosaved: {autosave}")

                if step % SAVE_EVERY == 0:
                    path = os.path.join(CHECKPOINT_DIR, f"step{step}.pt")
                    torch.save({"step": step, "model": model.state_dict(),
                                "opt": optimizer.state_dict(), "sched": scheduler.state_dict()}, path)
                    print(f"Saved checkpoint: {path}")

                pbar.update(1)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving emergency checkpoint...")
        path = os.path.join(CHECKPOINT_DIR, f"interrupted_step{step}.pt")
        torch.save({"step": step, "model": model.state_dict(),
                    "opt": optimizer.state_dict(), "sched": scheduler.state_dict()}, path)
        print(f"Saved: {path}")
        pbar.close()
        return

    final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
    torch.save({"step": step, "model": model.state_dict(),
                "opt": optimizer.state_dict(), "sched": scheduler.state_dict()}, final_path)
    print(f"\nDone. Final checkpoint: {final_path}")
    pbar.close()

if __name__ == "__main__":
    train()
