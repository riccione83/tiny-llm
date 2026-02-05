#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer for the from-scratch model.

Fixes:
- character_coverage < 1.0 to avoid overfitting rare scripts
- byte_fallback=True to handle any Unicode safely without dedicating vocab to it
- normalization_rule_name="nmt_nfkc" for consistent normalization
"""

import argparse
from pathlib import Path
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/pretrain.txt")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--model_prefix", default="tokenizer")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Missing input corpus: {inp}")

    Path(args.model_prefix).parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(inp),
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["User:", "Assistant:"],
    )

    print(f"Saved: {args.model_prefix}.model / {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
