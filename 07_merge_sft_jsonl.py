#!/usr/bin/env python3
"""
Merge multiple JSONL SFT datasets into one, with ratios and optional dedup.

Each input JSONL line must be:
  {"instruction": "...", "output": "..."}

Outputs:
  sft_merged.jsonl
"""

import argparse
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Set


def read_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            ins = row.get("instruction", "")
            outp = row.get("output", "")
            if not ins or not outp:
                continue
            out.append({"instruction": ins, "output": outp})
    return out


def fp(row: Dict) -> str:
    s = (row["instruction"] + "\n" + row["output"]).encode("utf-8", errors="ignore")
    return hashlib.sha1(s).hexdigest()


def sample_from(pool: List[Dict], k: int, rng: random.Random) -> List[Dict]:
    if not pool or k <= 0:
        return []
    # sample with replacement if needed
    if k <= len(pool):
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", required=True)
    ap.add_argument("--chat", required=True)
    ap.add_argument("--routing", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=400000)
    ap.add_argument("--summarize_ratio", type=float, default=0.70)
    ap.add_argument("--routing_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    summarize_path = Path(args.summarize)
    chat_path = Path(args.chat)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summarize = read_jsonl(summarize_path)
    chat = read_jsonl(chat_path)

    routing = []
    if args.routing:
        routing = read_jsonl(Path(args.routing))

    if not summarize:
        raise SystemExit("Empty summarize dataset")
    if not chat:
        raise SystemExit("Empty chat dataset")
    if args.routing and not routing:
        raise SystemExit("Routing path provided but dataset empty")

    # Allocate counts
    max_rows = int(args.max_rows)
    routing_ratio = float(args.routing_ratio) if args.routing else 0.0
    summarize_ratio = float(args.summarize_ratio)

    # ensure ratios sane
    summarize_ratio = max(0.0, min(1.0, summarize_ratio))
    routing_ratio = max(0.0, min(0.25, routing_ratio))
    chat_ratio = max(0.0, 1.0 - summarize_ratio - routing_ratio)

    n_sum = int(max_rows * summarize_ratio)
    n_routing = int(max_rows * routing_ratio)
    n_chat = max_rows - n_sum - n_routing

    merged = []
    merged += sample_from(summarize, n_sum, rng)
    merged += sample_from(chat, n_chat, rng)
    if routing:
        merged += sample_from(routing, n_routing, rng)

    rng.shuffle(merged)

    if args.dedup:
        seen: Set[str] = set()
        deduped = []
        for r in merged:
            h = fp(r)
            if h in seen:
                continue
            seen.add(h)
            deduped.append(r)
        merged = deduped

    with out_path.open("w", encoding="utf-8") as f:
        for r in merged[:max_rows]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {out_path} | rows={min(len(merged), max_rows):,}")
    print(f"Mix: summarize={n_sum:,} chat={n_chat:,} routing={n_routing:,} (ratios approx {summarize_ratio:.2f}/{chat_ratio:.2f}/{routing_ratio:.2f})")


if __name__ == "__main__":
    main()
