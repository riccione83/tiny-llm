#!/usr/bin/env python3
"""
Stream a public text dataset and write a cleaner plain-text corpus to disk.

"Do your magic" edition:
- Keeps it deterministic + streaming (no huge RAM)
- Removes/control characters and collapses whitespace
- Filters out non-English-heavy lines (prevents tokenizer learning lots of rare scripts)
- Skips ultra-short / low-letter-density blocks
- Tries to avoid wiki junk (very template-like / symbol-heavy lines)
- Writes double-newline separated document blocks

Default dataset: wikipedia 20220301.en (same as yours).
"""

import argparse
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ----------------------------
# Cleaning + filtering helpers
# ----------------------------

_ws_re = re.compile(r"[ \t]+")
_many_nl_re = re.compile(r"\n{3,}")
_bad_line_re = re.compile(r"^[\W_]{10,}$")  # line is mostly symbols
_url_re = re.compile(r"https?://\S+|www\.\S+")


def normalize_text(s: str) -> str:
    # Normalize unicode to reduce weird variants (fullwidth, compatibility chars)
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control chars except \n and \t
    s = "".join(ch for ch in s if (ch == "\n" or ch == "\t" or (ord(ch) >= 32 and unicodedata.category(ch) != "Cc")))
    s = _ws_re.sub(" ", s)
    s = _many_nl_re.sub("\n\n", s)
    return s.strip()


def ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    # Count printable-ish ASCII including newline
    good = 0
    total = 0
    for ch in s:
        if ch == "\n":
            continue
        total += 1
        o = ord(ch)
        if 32 <= o <= 126:
            good += 1
    return good / max(1, total)


def letter_ratio(s: str) -> float:
    if not s:
        return 0.0
    letters = 0
    total = 0
    for ch in s:
        if ch == "\n":
            continue
        total += 1
        if ch.isalpha():
            letters += 1
    return letters / max(1, total)


def looks_like_wiki_junk_line(line: str) -> bool:
    l = line.strip()
    if not l:
        return True
    # Lots of URLs / references-like
    if _url_re.search(l):
        return True
    # All symbols / separators
    if _bad_line_re.match(l):
        return True
    # Very template-ish / markup-ish fragments
    if l.startswith(("{|", "|}", "{{", "}}", "[[", "]]", "==", "*", "#", "|", "File:", "Image:", "Category:")):
        # these are common in raw wiki dumps; the HF wikipedia text is usually clean,
        # but keep this as a guard.
        return True
    return False


def filter_block(text: str, *, min_chars: int, min_ascii: float, min_letters: float) -> str | None:
    """
    Returns cleaned block or None if it should be skipped.
    """
    text = normalize_text(text)
    if len(text) < min_chars:
        return None

    # Drop blocks that are too non-ascii / too non-letter dense
    if ascii_ratio(text) < min_ascii:
        return None
    if letter_ratio(text) < min_letters:
        return None

    # Line-level cleanup: remove obvious junk lines, then re-join paragraphs
    lines = [ln.strip() for ln in text.split("\n")]
    kept = []
    for ln in lines:
        if not ln:
            kept.append("")  # keep paragraph breaks
            continue
        if looks_like_wiki_junk_line(ln):
            continue
        kept.append(ln)

    out = "\n".join(kept)
    out = _many_nl_re.sub("\n\n", out).strip()

    # Re-check after filtering
    if len(out) < min_chars:
        return None
    if ascii_ratio(out) < min_ascii:
        return None
    if letter_ratio(out) < min_letters:
        return None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pretrain.txt")
    ap.add_argument("--target_gb", type=float, default=5.0)
    ap.add_argument("--dataset", default="wikipedia")
    ap.add_argument("--config", default="20220301.en")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_key", default="text")

    # Cleaning knobs (tuned to reduce multilingual/script noise without being too aggressive)
    ap.add_argument("--min_chars", type=int, default=500, help="minimum characters per kept block")
    ap.add_argument("--min_ascii", type=float, default=0.92, help="min printable ASCII ratio (0-1)")
    ap.add_argument("--min_letters", type=float, default=0.55, help="min letter ratio (0-1)")

    # Determinism
    ap.add_argument("--max_docs", type=int, default=0, help="optional cap on number of documents (0 = no cap)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.target_gb * (1024**3))
    wrote = 0
    seen_docs = 0
    kept_docs = 0
    skipped_docs = 0

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    with out_path.open("w", encoding="utf-8") as f:
        pbar = tqdm(desc="writing", unit="B", total=target_bytes)
        for row in ds:
            seen_docs += 1
            raw = row.get(args.text_key) or ""
            cleaned = filter_block(
                raw,
                min_chars=args.min_chars,
                min_ascii=args.min_ascii,
                min_letters=args.min_letters,
            )
            if not cleaned:
                skipped_docs += 1
                continue

            block = cleaned + "\n\n"
            b = block.encode("utf-8")  # no ignore: we cleaned to safe unicode
            if wrote + len(b) > target_bytes:
                break

            f.write(block)
            wrote += len(b)
            kept_docs += 1
            pbar.update(len(b))

            if args.max_docs and kept_docs >= args.max_docs:
                break

        pbar.close()

    print(f"Saved: {out_path} ({wrote/1024**3:.2f} GB)")
    print(f"Docs seen={seen_docs:,} kept={kept_docs:,} skipped={skipped_docs:,}")
    print(f"Filters: min_chars={args.min_chars} min_ascii={args.min_ascii} min_letters={args.min_letters}")


if __name__ == "__main__":
    main()
