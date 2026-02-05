#!/usr/bin/env python3
"""
Create a small routing + format-constraints SFT dataset (JSONL).

This teaches:
- when to answer normally vs summarize
- hard output constraints ("number only", "YES/NO only")

Writes:
  data/routing_constraints_sft.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/routing_constraints_sft.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=30_000)
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Greetings routing
    greet = [
        ("User: Hi\nAssistant:", "Hi! How can I help?"),
        ("User: Hello\nAssistant:", "Hello! What can I do for you?"),
        ("User: Good morning!\nAssistant:", "Good morning! How can I help?"),
    ]
    rows += [{"instruction": i, "output": o} for i, o in greet]

    # Math: number-only
    math = [
        ("What is 20/5?", "4"),
        ("What is 7+8?", "15"),
        ("What is 9*6?", "54"),
        ("What is 100-37?", "63"),
        ("What is 12/3?", "4"),
        ("What is 2+2?", "4"),
    ]
    for q, ans in math:
        rows.append({
            "instruction": f"User: {q} Answer with a number only.\nAssistant:",
            "output": ans
        })

    # YES/NO only
    yesno = [
        ("Is 2+2=4?", "YES"),
        ("Is Paris in France?", "YES"),
        ("Is 10 greater than 50?", "NO"),
        ("Is water dry?", "NO"),
    ]
    for q, ans in yesno:
        rows.append({
            "instruction": f"User: Reply YES or NO only: {q}\nAssistant:",
            "output": ans
        })

    # Summarization routing: includes actual text
    summ = [
        ("The internet is a global network of networks that allows devices to communicate using standard protocols. It enables services like the web and email.",
         "The internet connects networks worldwide so devices can communicate using shared protocols. It powers services like the web and email and keeps expanding as more devices join."),
        ("Machine learning is a method where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each case.",
         "Machine learning lets computers learn patterns from data to make predictions or decisions. Instead of hand-coding rules, models improve by training on examples."),
    ]
    for txt, out in summ:
        rows.append({
            "instruction": f"User: Summarize in 2 sentences: {txt}\nAssistant:",
            "output": out
        })

    # replicate to reach args.n with random sampling
    base = rows[:]
    while len(rows) < args.n:
        rows.append(random.choice(base))

    random.shuffle(rows)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows[:args.n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {out_path} | rows={args.n:,}")


if __name__ == "__main__":
    main()
