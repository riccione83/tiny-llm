#!/usr/bin/env python3
"""
Pretokenize a text corpus into a numpy memmap array of int32 token ids.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/pretrain.txt")
    ap.add_argument("--tokenizer", default="tokenizer.model")
    ap.add_argument("--out", default="data/pretrain_tokens.npy")
    ap.add_argument("--dtype", default="int32")
    ap.add_argument("--chunk_lines", type=int, default=2048)
    args = ap.parse_args()

    inp = Path(args.input)
    tok = Path(args.tokenizer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Missing input corpus: {inp}")
    if not tok.exists():
        raise SystemExit(f"Missing tokenizer: {tok}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(tok))
    eos = sp.eos_id()

    # First pass: count tokens to size the memmap.
    # This can take a while on multi-GB corpora; show a progress bar so it doesn't look "stuck".
    print("Pass 1/2: counting tokens...")
    total_tokens = 0
    with inp.open("r", encoding="utf-8", errors="ignore") as f:
        buf = []
        for ln in tqdm(f, desc="counting", unit="lines"):
            buf.append(ln)
            if len(buf) >= args.chunk_lines:
                text = "".join(buf).strip()
                buf = []
                if not text:
                    continue
                ids = sp.encode(text, out_type=int)
                if eos is not None:
                    ids.append(eos)
                total_tokens += len(ids)
        if buf:
            text = "".join(buf).strip()
            if text:
                ids = sp.encode(text, out_type=int)
                if eos is not None:
                    ids.append(eos)
                total_tokens += len(ids)

    print(f"Pass 1/2 done. tokens={total_tokens:,}")

    # Create a .npy-backed memmap.
    # We store as flat 1D array.
    arr = np.lib.format.open_memmap(
        str(out),
        mode="w+",
        dtype=np.dtype(args.dtype),
        shape=(total_tokens,),
    )

    # Second pass: write.
    print("Pass 2/2: writing token ids...")
    idx = 0
    with inp.open("r", encoding="utf-8", errors="ignore") as f:
        buf = []
        for ln in tqdm(f, desc="tokenizing"):
            buf.append(ln)
            if len(buf) >= args.chunk_lines:
                text = "".join(buf).strip()
                buf = []
                if not text:
                    continue
                ids = sp.encode(text, out_type=int)
                if eos is not None:
                    ids.append(eos)
                arr[idx:idx+len(ids)] = ids
                idx += len(ids)
        if buf:
            text = "".join(buf).strip()
            if text:
                ids = sp.encode(text, out_type=int)
                if eos is not None:
                    ids.append(eos)
                arr[idx:idx+len(ids)] = ids
                idx += len(ids)

    if idx != total_tokens:
        raise RuntimeError(f"write mismatch: wrote={idx} expected={total_tokens}")

    arr.flush()
    print(f"Saved: {out} | tokens={total_tokens:,}")


if __name__ == "__main__":
    main()
