#!/usr/bin/env python3
"""
Build a summarization SFT dataset (JSONL) in chat format, token-length safe.

Output JSONL rows:
  {"instruction": "User: ...\nAssistant:", "output": "<summary>"}

Key features:
- Chat-style prompts using "User:" / "Assistant:"
- Token-length filtering and optional truncation to fit a target max_tokens
- Mix of instruction templates for robustness
- Can include multiple splits (train/validation/test) to get more rows

Example:
  python 04_build_summarize_sft_chat.py ^
    --out data/summarize_sft.jsonl ^
    --tokenizer tokenizer.model ^
    --max 600000 ^
    --splits train validation test ^
    --block_size 768 ^
    --max_tokens 760 ^
    --truncate_input
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm


def normalize(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def to_two_sentences(summary: str) -> str:
    summary = (summary or "").replace("\n", " ")
    sents = split_sentences(summary)
    if not sents:
        return ""
    if len(sents) == 1:
        return sents[0]
    return " ".join(sents[:2]).strip()


def make_prompt(template: str, text: str) -> str:
    # Chat-style, consistent with your LoRA trainer
    return f"User: {template}\n{text}\nAssistant:"


def maybe_add_web_noise(text: str) -> str:
    # Tiny “website-like paste” noise. Keeps it lightweight and safe.
    # This helps later when users paste messy pages.
    return (
        "Home | About | Contact\n\n"
        "### Content\n\n"
        + text
        + "\n\n• Related\n• Privacy Policy"
    )


def token_len(sp: spm.SentencePieceProcessor, s: str, add_eos: bool = True) -> int:
    ids = sp.encode(s, out_type=int)
    if add_eos and sp.eos_id() is not None:
        ids.append(sp.eos_id())
    return len(ids)


def truncate_text_to_fit(
    sp: spm.SentencePieceProcessor,
    template: str,
    text: str,
    summary: str,
    max_tokens: int,
) -> Optional[Tuple[str, str]]:
    """
    Try to truncate *text* (the document) so that:
      encode(prompt + " " + summary + eos) <= max_tokens

    Returns (new_text, summary) or None if it can't fit even with heavy truncation.
    """
    text = normalize(text)
    summary = normalize(summary)

    # Keep summary fixed (it's already short)
    summary = to_two_sentences(summary)
    if not text or not summary:
        return None

    # Prefix/suffix tokens (without the document text)
    prefix = f"User: {template}\n"
    suffix = "\nAssistant:"

    prefix_ids = sp.encode(prefix, out_type=int)
    suffix_ids = sp.encode(suffix, out_type=int)
    summary_ids = sp.encode(" " + summary, out_type=int)

    eos_extra = 1 if sp.eos_id() is not None else 0

    # Budget for the document text tokens
    budget = max_tokens - (len(prefix_ids) + len(suffix_ids) + len(summary_ids) + eos_extra)

    # If even zero text can't fit, give up
    if budget <= 8:
        return None

    text_ids = sp.encode(text, out_type=int)
    if len(text_ids) <= budget:
        return text, summary

    # Truncate document tokens to budget and decode back
    text_ids = text_ids[:budget]
    new_text = sp.decode(text_ids).strip()
    if not new_text:
        return None
    return new_text, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/summarize_sft.jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max", type=int, default=600_000, help="max rows to write total")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_chars", type=int, default=240)
    ap.add_argument("--max_chars", type=int, default=4000)

    ap.add_argument("--splits", nargs="+", default=["train"], help="e.g. train validation test")

    ap.add_argument("--block_size", type=int, default=768)
    ap.add_argument("--max_tokens", type=int, default=760, help="token cap for (prompt+output+eos)")
    ap.add_argument("--truncate_input", action="store_true", help="truncate document to fit max_tokens instead of skipping")

    ap.add_argument("--web_noise_prob", type=float, default=0.10, help="probability to add small web-like noise")

    args = ap.parse_args()
    random.seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    # Prompt templates for robustness
    templates = [
        "Summarize in 2 sentences:",
        "Summarize:",
        "Please summarize this text:",
    ]

    wrote = 0
    skipped_len = 0
    skipped_empty = 0
    skipped_filter = 0

    # For safety: keep inside block_size+1 used by training
    # We store sequences of length <= block_size+1 in the LoRA dataset code;
    # here we cap prompt+output+eos by max_tokens, where max_tokens <= block_size+1.
    hard_cap = min(args.max_tokens, args.block_size + 1)

    def emit(doc_text: str, ref_summary: str):
        nonlocal wrote, skipped_len, skipped_empty, skipped_filter

        if wrote >= args.max:
            return

        doc_text = normalize(doc_text)
        ref_summary = normalize(ref_summary)
        ref_summary = to_two_sentences(ref_summary)

        if not doc_text or not ref_summary:
            skipped_empty += 1
            return

        # Basic char filters first (cheap)
        if not (args.min_chars <= len(doc_text) <= args.max_chars):
            skipped_filter += 1
            return

        if "as an ai language model" in ref_summary.lower():
            skipped_filter += 1
            return

        # Optional web-like noise
        if args.web_noise_prob > 0 and random.random() < args.web_noise_prob:
            doc_text2 = maybe_add_web_noise(doc_text)
        else:
            doc_text2 = doc_text

        template = random.choice(templates)

        # Token-fit logic
        if args.truncate_input:
            out_pair = truncate_text_to_fit(sp, template, doc_text2, ref_summary, hard_cap)
            if out_pair is None:
                skipped_len += 1
                return
            doc_text2, ref_summary2 = out_pair
        else:
            prompt = make_prompt(template, doc_text2)
            full = normalize(prompt + " " + ref_summary)
            if token_len(sp, full, add_eos=True) > hard_cap:
                skipped_len += 1
                return
            ref_summary2 = ref_summary

        prompt = make_prompt(template, doc_text2)
        row = {"instruction": prompt, "output": ref_summary2}
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        wrote += 1

    def iter_dataset(ds, n: int) -> Iterable[int]:
        idxs = list(range(n))
        random.shuffle(idxs)
        return idxs

    with out_path.open("w", encoding="utf-8") as f_out:
        # CNN/DailyMail
        for split in args.splits:
            if wrote >= args.max:
                break
            print(f"Loading cnn_dailymail 3.0.0 [{split}]...")
            ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
            for i in tqdm(iter_dataset(ds, len(ds)), desc=f"cnn_dailymail:{split}"):
                if wrote >= args.max:
                    break
                row = ds[int(i)]
                emit(row.get("article", ""), row.get("highlights", ""))

        # XSum
        for split in args.splits:
            if wrote >= args.max:
                break
            print(f"Loading xsum [{split}]...")
            ds = load_dataset("xsum", split=split)
            for i in tqdm(iter_dataset(ds, len(ds)), desc=f"xsum:{split}"):
                if wrote >= args.max:
                    break
                row = ds[int(i)]
                emit(row.get("document", ""), row.get("summary", ""))

    print(f"Saved: {out_path} | rows={wrote:,}")
    print(f"Skipped: too_long={skipped_len:,} empty={skipped_empty:,} filtered={skipped_filter:,}")
    print(f"Settings: block_size={args.block_size} max_tokens={hard_cap} truncate_input={args.truncate_input} splits={args.splits}")


if __name__ == "__main__":
    main()
