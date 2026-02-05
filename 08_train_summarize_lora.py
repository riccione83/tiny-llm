#!/usr/bin/env python3
"""
Train a LoRA adapter (summarization + basic chat) on top of a base checkpoint.

IMPORTANT:
- Expects JSONL rows like:
    {"instruction": "User: ...\\nAssistant:", "output": "..."}
  i.e. instruction is ALREADY a full chat-style prompt.
- We compute a prefix mask so loss is applied ONLY to the assistant output.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Must match your base model config
BLOCK_SIZE = 768
EMBED_DIM = 896
NUM_HEADS = 14
NUM_LAYERS = 16
DROPOUT = 0.0

IGNORE_INDEX = -100
PAD_ID = 0  # matches your SentencePiece pad_id=0


def normalize(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------- Model (same as base) ----------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack((out1, out2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=q.dtype)
        q = self.rope.apply_rotary(q, cos, sin)
        k = self.rope.apply_rotary(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        hidden = int((dim * 8) / 3)
        hidden = (hidden + 255) // 256 * 256
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FastGPT(nn.Module):
    def __init__(self, vocab, dim, heads, layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, dim)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        if x.shape[1] > self.block_size:
            x = x[:, -self.block_size:]
        h = self.drop(self.tok(x))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm_f(h))


# ---------------- LoRA injection ----------------

LORA_R = 16
LORA_ALPHA = 32
LORA_SCALE = LORA_ALPHA / LORA_R
LORA_TARGET = ["attn.qkv", "attn.proj"]


def inject_lora(model: nn.Module, device: str):
    injected = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in LORA_TARGET):
            in_f, out_f = module.in_features, module.out_features

            module.lora_A = nn.Parameter(torch.zeros(LORA_R, in_f, device=device))
            module.lora_B = nn.Parameter(torch.zeros(out_f, LORA_R, device=device))
            module.lora_scale = float(LORA_SCALE)

            nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B)

            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False

            orig_forward = module.forward

            def forward(x, orig_forward=orig_forward, m=module):
                return orig_forward(x) + ((x @ m.lora_A.T) @ m.lora_B.T) * m.lora_scale

            module.forward = forward
            injected += 1

    print(f"LoRA injected: {injected}", flush=True)


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            sd[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            sd[f"{name}.lora_scale"] = torch.tensor(float(getattr(module, "lora_scale", LORA_SCALE)))
    return sd


def load_base_checkpoint(model: nn.Module, path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)


# ---------------- Dataset ----------------

def split_prefix_response(
    tokenizer: spm.SentencePieceProcessor,
    prompt_text: str,   # already "User: ...\nAssistant:"
    output_text: str
) -> Tuple[List[int], int]:
    """
    Returns:
      full_ids: token ids for prompt + ' ' + output + eos
      prefix_len: number of tokens belonging to the prompt (used to mask loss)
    """
    eos = tokenizer.eos_id()
    prompt_text = normalize(prompt_text)
    output_text = normalize(output_text)

    prefix_ids = tokenizer.encode(prompt_text, out_type=int)

    full_text = prompt_text + " " + output_text
    full_ids = tokenizer.encode(full_text, out_type=int)

    if eos is not None:
        full_ids.append(eos)

    return full_ids, len(prefix_ids)


class JsonlSFTDataset(Dataset):
    """
    Streaming JSONL dataset with byte offsets.

    Produces:
      x: [block_size]
      y: [block_size] with IGNORE_INDEX for prompt tokens + pad tokens
    """

    def __init__(self, jsonl_path: str, tokenizer: spm.SentencePieceProcessor, block_size: int):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.offsets: List[int] = []

        kept = 0
        skipped_long = 0
        skipped_bad = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped_bad += 1
                    continue

                instr = normalize(row.get("instruction", ""))
                out = normalize(row.get("output", ""))
                if not instr or not out:
                    skipped_bad += 1
                    continue

                ids, _ = split_prefix_response(self.tokenizer, instr, out)

                # Need <= block_size+1 tokens so we can shift into x/y
                if len(ids) > (block_size + 1):
                    skipped_long += 1
                    continue

                self.offsets.append(pos)
                kept += 1

                if kept % 20000 == 0:
                    print(f"  indexed {kept:,} (skipped too-long {skipped_long:,}, bad {skipped_bad:,})", flush=True)

        print(f"SFT samples kept: {kept:,} | skipped too-long: {skipped_long:,} | skipped bad: {skipped_bad:,}", flush=True)
        self._file = None

    def __len__(self):
        return len(self.offsets)

    def _get_file(self):
        if self._file is None:
            self._file = open(self.jsonl_path, "r", encoding="utf-8")
        return self._file

    def __getitem__(self, idx):
        f = self._get_file()
        f.seek(self.offsets[idx])
        row = json.loads(f.readline())

        instr = normalize(row.get("instruction", ""))
        out = normalize(row.get("output", ""))

        ids, prefix_len = split_prefix_response(self.tokenizer, instr, out)

        # Pad/truncate to block_size+1
        if len(ids) < (self.block_size + 1):
            ids = ids + [PAD_ID] * ((self.block_size + 1) - len(ids))
        ids = ids[: (self.block_size + 1)]

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        # Mask loss on prompt tokens:
        # If prefix_len tokens are prompt, first (prefix_len-1) targets correspond to prompt tokens.
        mask_upto = max(0, prefix_len - 1)
        y[:mask_upto] = IGNORE_INDEX

        # Mask loss on pad tokens
        y[y == PAD_ID] = IGNORE_INDEX
        return x, y


# ---------------- Sampling (optional) ----------------

@torch.no_grad()
def _sample_top_p(probs: torch.Tensor, top_p: float) -> int:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    cutoff = torch.searchsorted(cumsum, torch.tensor(top_p, device=probs.device))
    cutoff = int(cutoff.item())
    cutoff = max(1, min(cutoff + 1, sorted_probs.numel()))
    filtered_probs = sorted_probs[:cutoff]
    filtered_idx = sorted_idx[:cutoff]
    filtered_probs = filtered_probs / filtered_probs.sum()
    next_i = torch.multinomial(filtered_probs, 1).item()
    return int(filtered_idx[next_i].item())


@torch.no_grad()
def generate(
    sp: spm.SentencePieceProcessor,
    model: nn.Module,
    device: str,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    model.eval()
    ids = sp.encode(normalize(prompt), out_type=int)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    eos = sp.eos_id()

    for _ in range(max_new_tokens):
        logits = model(x[:, -BLOCK_SIZE:])[:, -1, :] / max(1e-6, float(temperature))
        probs = F.softmax(logits, dim=-1).squeeze(0)
        nxt = _sample_top_p(probs, float(top_p))
        x = torch.cat([x, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        if eos is not None and nxt == eos:
            break

    text = sp.decode(x[0].tolist())
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1]
    return text.strip()


# ---------------- Train ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--out_dir", default="finetuning")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--sample_every", type=int, default=200, help="0 to disable")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--resume_model_only",
        action="store_true",
        help="With --resume, load only model weights from autosave and reset optimizer/scheduler state.",
    )
    ap.add_argument(
        "--max_opt_steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps (0 = no limit). Useful for safe micro-retrain.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Loading tokenizer...", flush=True)
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab = sp.get_piece_size()
    print(f"Vocab: {vocab:,} | eos_id={sp.eos_id()}", flush=True)

    print("Loading SFT dataset...", flush=True)
    ds = JsonlSFTDataset(args.sft_jsonl, sp, BLOCK_SIZE)
    if len(ds) == 0:
        raise SystemExit("No SFT samples kept. Check dataset builder and token length limits.")

    print("Creating dataloader...", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print("Building model...", flush=True)
    model = FastGPT(vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(device)

    print("Loading base checkpoint...", flush=True)
    load_base_checkpoint(model, args.base_ckpt, device)

    # Freeze base weights
    for p in model.parameters():
        p.requires_grad = False

    inject_lora(model, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda") else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / "lora_adapter.pt"
    full_path = out_dir / "lora_full_state.pt"
    autosave_path = out_dir / "lora_autosave.pt"

    step = 0
    start_epoch = 0
    if args.resume and autosave_path.exists():
        ckpt = torch.load(str(autosave_path), map_location=device)
        model.load_state_dict(ckpt["model"])
        if args.resume_model_only:
            step = 0
            start_epoch = 0
            print(
                f"Resumed MODEL only from {autosave_path}; optimizer/scheduler reset (lr={args.lr:.2e}).",
                flush=True,
            )
        else:
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            step = int(ckpt.get("step", 0))
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"Resumed from {autosave_path} at opt_step {step}, epoch {start_epoch+1}", flush=True)
    elif args.resume:
        print(f"Resume requested but no autosave at {autosave_path}; starting fresh.", flush=True)

    def save_autosave(epoch_idx: int, step_val: int):
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step_val, "epoch": epoch_idx},
            str(autosave_path),
        )

    def atomic_save(path: Path, obj):
        tmp = Path(str(path) + ".tmp")
        torch.save(obj, str(tmp))
        os.replace(str(tmp), str(path))

    model.train()
    stop_early = False
    try:
        for epoch in range(start_epoch, args.epochs):
            pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
            loss_acc = 0.0
            n_acc = 0

            for i, (x, y) in enumerate(pbar):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if (i % args.grad_accum) == 0:
                    opt.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=IGNORE_INDEX)
                        loss = loss / args.grad_accum
                    scaler.scale(loss).backward()
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=IGNORE_INDEX)
                    loss = loss / args.grad_accum
                    loss.backward()

                if ((i + 1) % args.grad_accum) == 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        opt.step()

                    sched.step()
                    step += 1

                    loss_acc += float(loss.item()) * args.grad_accum
                    n_acc += 1

                    if step % args.print_every == 0 and n_acc > 0:
                        avg = loss_acc / n_acc
                        ppl = math.exp(min(avg, 20))
                        pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.2f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                        loss_acc = 0.0
                        n_acc = 0

                    if args.save_every > 0 and (step % args.save_every) == 0:
                        save_autosave(epoch, step)

                    if args.sample_every > 0 and (step % args.sample_every == 0):
                        prompts = [
                            "User: Hi\nAssistant:",
                            "User: What is 20/5? Answer with a number only.\nAssistant:",
                            "User: Summarize in 2 sentences: The internet is a global network of networks that allows devices to communicate using standard protocols. It enables services like the web and email.\nAssistant:",
                            "User: Summarize in exactly 2 sentences and do not add any information not present in the text: She was studying alone but suddenly her friend knocked at the door. She got distracted and finished only half of her homework.\nAssistant:",
                            "User: Cosa dice questo articolo? Home | News | Contact. Published: 2026-01-18. NVIDIA announced RTX 6090 and RTX 6080. RTX 6090 launches first, RTX 6080 two weeks later.\nAssistant:",
                        ]
                        model.eval()
                        with torch.no_grad():
                            for ptxt in prompts:
                                out = generate(sp, model, device, ptxt, max_new_tokens=120, temperature=0.0, top_p=1.0)
                                print("\n--- SAMPLE ---")
                                print(f"Prompt: {ptxt.splitlines()[0]}")
                                print(f"Output: {out}\n")
                        model.train()

                    if args.max_opt_steps > 0 and step >= args.max_opt_steps:
                        print(f"Reached max_opt_steps={args.max_opt_steps}; stopping early.", flush=True)
                        save_autosave(epoch, step)
                        stop_early = True
                        break

            if stop_early:
                break

            save_autosave(epoch, step)

    except KeyboardInterrupt:
        print("\nInterrupted, saving autosave...")
        save_autosave(epoch if "epoch" in locals() else start_epoch, step)
        print(f"Saved autosave: {autosave_path}")
        return

    atomic_save(adapter_path, lora_state_dict(model))
    atomic_save(full_path, model.state_dict())
    print(f"Saved: {adapter_path}")
    print(f"Saved: {full_path}")


if __name__ == "__main__":
    main()
