#!/usr/bin/env python3
"""
From-scratch base pretraining (v2 "modern" blocks, ~184M):

- RoPE (rotary positional embeddings)
- RMSNorm
- SwiGLU MLP
- SDPA attention (causal)

Upgrades (minimal, high ROI):
- Train/val split on token memmap
- Periodic eval of train/val loss
- Warmup + cosine decay LR schedule
- Resume saves extra metadata (val split + args)

Windows + single GPU friendly: no multiprocessing; predictable memory.
"""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Model config (~184M). BLOCK_SIZE can be overridden by CLI.
BLOCK_SIZE = 512
EMBED_DIM = 896
NUM_HEADS = 14
NUM_LAYERS = 16
DROPOUT = 0.0


def sample_batch(tokens_cpu: torch.Tensor, max_i: int, batch_size: int, block_size: int, device: str):
    """
    Fast batch sampler without Python loops.

    tokens_cpu: 1D CPU tensor (typically backed by np.memmap).
    max_i: len(tokens_cpu) - (block_size + 1)
    """
    ix = torch.randint(0, max_i, (batch_size,), device="cpu")

    if not hasattr(sample_batch, "_offsets") or sample_batch._offsets.numel() != (block_size + 1):
        sample_batch._offsets = torch.arange(block_size + 1, device="cpu")
    offsets = sample_batch._offsets

    idx = ix[:, None] + offsets[None, :]  # [B, T+1]
    batch = tokens_cpu[idx]  # still on CPU
    batch = batch.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))

    x = batch[:, :-1].contiguous()
    y = batch[:, 1:].contiguous()
    return x, y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class RoPE(nn.Module):
    """Rotary embeddings (GPT-NeoX style)."""
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        self._dtype_cached = None
        self._device_cached = None

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._dtype_cached != dtype
            or self._device_cached != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            self._cos_cached = freqs.cos().to(dtype=dtype)
            self._sin_cached = freqs.sin().to(dtype=dtype)
            self._seq_len_cached = seq_len
            self._dtype_cached = dtype
            self._device_cached = device
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        out = torch.stack((out1, out2), dim=-1).flatten(-2)
        return out


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
    def __init__(self, dim: int, heads: int):
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


class FastGPTv2(nn.Module):
    def __init__(self, vocab: int, dim: int, heads: int, layers: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, dim)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        if x.shape[1] > self.block_size:
            x = x[:, -self.block_size:]
        h = self.drop(self.tok(x))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm_f(h))


@torch.no_grad()
def _sample_top_p(probs: torch.Tensor, top_p: float) -> int:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    cutoff = torch.searchsorted(cumsum, torch.tensor(top_p, device=probs.device))
    cutoff = int(cutoff.item())
    cutoff = max(1, min(cutoff + 1, sorted_probs.numel()))
    filtered_probs = sorted_probs[:cutoff]
    filtered_probs = filtered_probs / filtered_probs.sum()
    filtered_idx = sorted_idx[:cutoff]
    next_i = torch.multinomial(filtered_probs, 1).item()
    return int(filtered_idx[next_i].item())


@torch.no_grad()
def generate(
    sp: spm.SentencePieceProcessor,
    model: nn.Module,
    device: str,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> str:
    model.eval()
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    eos = sp.eos_id()
    for _ in range(max_new_tokens):
        logits = model(x[:, -BLOCK_SIZE:])[:, -1, :] / max(1e-6, float(temperature))
        probs = F.softmax(logits, dim=-1).squeeze(0)
        nxt = _sample_top_p(probs, float(top_p))
        x = torch.cat([x, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        if eos is not None and nxt == eos:
            break
    return sp.decode(x[0].tolist())


def _make_splits(tokens_cpu: torch.Tensor, block_size: int, val_frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split tokens into train/val contiguous chunks.
    Use the last val_frac as validation.

    Ensures each split can sample at least one block.
    """
    n = tokens_cpu.numel()
    if val_frac <= 0:
        return tokens_cpu, tokens_cpu[:0]

    val_len = int(n * val_frac)
    # Ensure val has room for at least one (block_size+1) sequence
    min_len = block_size + 2
    val_len = max(min_len, val_len)
    val_len = min(n - min_len, val_len) if n > 2 * min_len else min_len

    train = tokens_cpu[: n - val_len]
    val = tokens_cpu[n - val_len :]
    return train, val


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    vocab: int,
    tokens_train: torch.Tensor,
    tokens_val: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    iters: int,
) -> Tuple[float, float]:
    """
    Returns (train_loss, val_loss) estimated over 'iters' batches each.
    Uses the same random window sampling.
    """
    model.eval()

    def _avg_loss(tok: torch.Tensor) -> float:
        if tok.numel() < block_size + 2:
            return float("nan")
        max_i = tok.numel() - (block_size + 1)
        acc = 0.0
        for _ in range(iters):
            x, y = sample_batch(tok, max_i, batch_size, block_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
            acc += float(loss.item())
        return acc / max(1, iters)

    tr = _avg_loss(tokens_train)
    va = _avg_loss(tokens_val) if tokens_val.numel() > 0 else float("nan")
    model.train()
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/pretrain_tokens.npy")
    ap.add_argument("--tokenizer", default="tokenizer.model")
    ap.add_argument("--out_dir", default="checkpoints_v2")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--block_size", type=int, default=768, help="sequence length")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)

    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1, help="cosine decay target as a fraction of lr (e.g. 0.1)")

    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--sample_every", type=int, default=1000)

    ap.add_argument("--val_frac", type=float, default=0.001, help="fraction of tokens reserved for validation (e.g. 0.001 = 0.1%)")
    ap.add_argument("--eval_every", type=int, default=500, help="run eval every N *optimizer steps* (not micro-steps)")
    ap.add_argument("--eval_iters", type=int, default=50, help="batches for each loss estimate")

    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Allow block size override
    global BLOCK_SIZE
    BLOCK_SIZE = int(args.block_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    autosave = out_dir / "autosave.pt"
    final = out_dir / "final.pt"

    def _atomic_save(path: Path, obj: dict):
        tmp = Path(str(path) + ".tmp")
        torch.save(obj, str(tmp))
        os.replace(str(tmp), str(path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.tokenizer))
    vocab = sp.get_piece_size()
    print(f"Vocab: {vocab:,} | eos_id={sp.eos_id()}")

    tokens_np = np.load(args.tokens, mmap_mode="r")
    print(f"Tokens: {len(tokens_np):,} | dtype={tokens_np.dtype}")

    tokens_cpu_all = torch.from_numpy(tokens_np)
    tokens_train, tokens_val = _make_splits(tokens_cpu_all, BLOCK_SIZE, args.val_frac)
    print(f"Split: train_tokens={tokens_train.numel():,} | val_tokens={tokens_val.numel():,} | val_frac={args.val_frac}")

    max_i_train = tokens_train.numel() - (BLOCK_SIZE + 1)
    if max_i_train <= 0:
        raise SystemExit("Not enough tokens to sample one training sequence. Increase corpus/tokens or lower block_size.")

    model = FastGPTv2(vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Scheduler: warmup then cosine decay to min_lr_ratio
    def lr_lambda(step: int) -> float:
        # step here = scheduler step calls count (we call sched.step() per optimizer step)
        if step < args.warmup:
            return float(step + 1) / max(1, args.warmup)
        # cosine from 1.0 down to min_lr_ratio over remaining optimizer steps
        total_opt_steps = max(1, args.steps // max(1, args.grad_accum))
        done = min(step - args.warmup, max(1, total_opt_steps - args.warmup))
        t = done / max(1, (total_opt_steps - args.warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    sched = LambdaLR(opt, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # Resume
    step0 = 0
    opt_step0 = 0
    if args.resume and autosave.exists():
        ckpt = torch.load(str(autosave), map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step0 = int(ckpt.get("step", 0))
        opt_step0 = int(ckpt.get("opt_step", 0))
        print(f"Resumed from {autosave} at micro-step {step0} (opt_step {opt_step0})")
    elif args.resume:
        print(f"Resume requested but no autosave found at {autosave}; starting from scratch.")

    model.train()
    t0 = time.time()
    pbar = tqdm(range(step0, args.steps), desc="train_v2")
    last_step = step0

    # helper: compute current optimizer step number from micro-step
    def micro_to_opt_step(micro_step: int) -> int:
        # optimizer step happens when (micro_step+1) % grad_accum == 0
        return (micro_step + 1) // max(1, args.grad_accum)

    try:
        for step in pbar:
            last_step = step + 1

            if (step % args.grad_accum) == 0:
                opt.zero_grad(set_to_none=True)

            x, y = sample_batch(tokens_train, max_i_train, args.batch_size, BLOCK_SIZE, device)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1)) / args.grad_accum
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1)) / args.grad_accum
                loss.backward()

            did_opt_step = False
            if ((step + 1) % args.grad_accum) == 0:
                did_opt_step = True
                if scaler is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                opt_step0 += 1

            # Pretty stats every print_every micro-steps
            if (step + 1) % args.print_every == 0:
                if device == "cuda":
                    torch.cuda.synchronize()
                l = float(loss.item() * args.grad_accum)
                ppl = math.exp(min(l, 20))
                dt = time.time() - t0
                it_s = args.print_every / max(1e-9, dt)
                tok_s = it_s * args.batch_size * BLOCK_SIZE
                opt_s = it_s / max(1, args.grad_accum)
                pbar.set_postfix(
                    loss=f"{l:.4f}",
                    ppl=f"{ppl:.2f}",
                    lr=f"{sched.get_last_lr()[0]:.2e}",
                    it_s=f"{it_s:.2f}",
                    opt_s=f"{opt_s:.2f}",
                    tok_s=f"{tok_s/1e6:.2f}M",
                )
                t0 = time.time()

            # Autosave
            if (step + 1) % args.save_every == 0:
                _atomic_save(
                    autosave,
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "step": step + 1,           # micro-step
                        "opt_step": opt_step0,      # optimizer step count
                        "block_size": BLOCK_SIZE,
                        "val_frac": args.val_frac,
                    },
                )

            # Eval every N optimizer steps (not micro-steps)
            if did_opt_step and args.eval_every > 0 and (opt_step0 % args.eval_every == 0):
                tr, va = estimate_loss(
                    model,
                    vocab,
                    tokens_train,
                    tokens_val,
                    BLOCK_SIZE,
                    args.batch_size,
                    device,
                    args.eval_iters,
                )
                print(f"\n[EVAL opt_step={opt_step0}] train_loss={tr:.4f} | val_loss={va:.4f}")

            # Sample generation (still based on micro-step, like before)
            if args.sample_every > 0 and ((step + 1) % args.sample_every == 0):
                prompts = [
                    "Machine learning is",
                    "The Internet is",
                    "France is a country in",
                    "In mathematics,",
                    "History",
                ]
                print("\n--- SAMPLE ---")
                for p in prompts:
                    txt = generate(sp, model, device, p, max_new_tokens=100, temperature=0.9, top_p=0.95)
                    out = txt.split("Assistant:")[-1].strip() if "Assistant:" in txt else txt.strip()
                    print(f"Prompt: {p.splitlines()[0]}")
                    print(f"Output: {out}\n")
                model.train()

    except KeyboardInterrupt:
        print(f"\nInterrupted. Saving autosave checkpoint at micro-step {last_step} (opt_step {opt_step0})...")
        _atomic_save(
            autosave,
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": last_step,
                "opt_step": opt_step0,
                "block_size": BLOCK_SIZE,
                "val_frac": args.val_frac,
            },
        )
        print(f"Saved: {autosave}")
        return

    _atomic_save(final, {"model": model.state_dict()})
    print(f"Saved final: {final}")


if __name__ == "__main__":
    main()
