#!/usr/bin/env python3
"""
Download a base HF causal LM model locally (model + tokenizer).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str) -> torch.dtype | None:
    n = (name or "auto").strip().lower()
    if n == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if n in {"fp16", "float16"}:
        return torch.float16
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a base model for full fine-tuning")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--output_dir", default="models/base")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))

    print("Base model downloaded.")
    print(f"- model_id:   {args.model_id}")
    print(f"- output_dir: {out.resolve()}")


if __name__ == "__main__":
    main()
