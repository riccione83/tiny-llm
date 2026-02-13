#!/usr/bin/env python3
"""
Merge a LoRA checkpoint into the base model and export a merged HF model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str) -> torch.dtype:
    n = (name or "auto").strip().lower()
    if n == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if n in {"float16", "fp16"}:
        return torch.float16
    if n in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if n in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_causal_lm(
    model_id_or_path: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    device_map: str,
):
    kwargs = {"trust_remote_code": bool(trust_remote_code), "device_map": device_map}
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            **kwargs,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a LoRA checkpoint and export merged HF model")
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--adapter_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto", help="auto|cpu|cuda|balanced...")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    base_model_dir = Path(args.base_model_dir).resolve()
    adapter_checkpoint = Path(args.adapter_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_model_dir.exists():
        raise SystemExit(f"Base model dir not found: {base_model_dir}")
    if not adapter_checkpoint.exists():
        raise SystemExit(f"Adapter checkpoint not found: {adapter_checkpoint}")

    dtype = resolve_dtype(args.dtype)

    print(f"Loading tokenizer from: {base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_dir),
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = int(tokenizer.eos_token_id)

    print(f"Loading base model: {base_model_dir}")
    model = load_causal_lm(
        model_id_or_path=str(base_model_dir),
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=str(args.device_map),
    )
    model.eval()

    print(f"Loading LoRA adapter checkpoint: {adapter_checkpoint}")
    peft_model = PeftModel.from_pretrained(model, str(adapter_checkpoint))
    peft_model.eval()
    if not hasattr(peft_model, "merge_and_unload"):
        raise SystemExit("Adapter merge is not supported: merge_and_unload() not available")

    print("Merging adapter into base model...")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    info = {
        "base_model_dir": str(base_model_dir),
        "adapter_checkpoint": str(adapter_checkpoint),
        "output_dir": str(output_dir),
        "dtype": str(dtype),
        "device_map": str(args.device_map),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "merge_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("Merged export complete.")
    print(f"- output_dir: {output_dir}")


if __name__ == "__main__":
    main()

