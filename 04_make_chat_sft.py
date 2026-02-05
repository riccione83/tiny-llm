#!/usr/bin/env python3
"""
Create a basic chat SFT dataset (JSONL) for greetings + small talk + basic help,
WITHOUT pushing summarization as default behavior.

Output schema (JSONL):
  {"instruction": "User: ...\nAssistant:", "output": "..."}

Writes:
  data/basic_chat_sft.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path


def norm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/basic_chat_sft.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=120_000, help="approx number of rows to generate")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Templates (NO "paste text" default) ---
    greetings = [
        ("Hi", "Hi! How can I help today?"),
        ("Hello", "Hello! What can I do for you?"),
        ("Hey", "Hey! What’s up?"),
        ("Good morning!", "Good morning! How can I help?"),
        ("Good evening!", "Good evening! What do you need?"),
        ("hiya", "Hi! What can I help with?"),
        ("hey there", "Hey there! How can I help?"),
    ]

    how_are_you = [
        ("How are you?", "I’m doing well, thanks! How can I help?"),
        ("how are you doing today?", "All good — ready to help. What do you need?"),
        ("How do you feel today?", "I’m here and ready. What are you working on?"),
    ]

    thanks = [
        ("Thanks!", "You’re welcome! Anything else you need?"),
        ("Thank you", "No problem — happy to help."),
        ("Nice one, thanks 🙂", "Glad it helped! Want to do the next step together?"),
        ("cheers", "Anytime! What’s next?"),
    ]

    what_can_you_do = [
        ("What can you do?", "I can answer questions, explain concepts, and summarize text when you ask. What would you like to do?"),
        ("How do I use you?", "Ask a question or tell me what you want (e.g., “summarize this”, “explain this”, “draft this”)."),
        ("Help me figure this out.", "Sure — tell me what you’re trying to achieve and what you’ve tried so far."),
    ]

    # small talk / friendly follow-ups
    followups = [
        ("Talk soon 🙂", "See you! 👋"),
        ("ok", "Great — what’s the next step?"),
        ("cool", "Nice. What would you like to do next?"),
    ]

    # Build pool
    pool = []
    pool += greetings
    pool += how_are_you
    pool += thanks
    pool += what_can_you_do
    pool += followups

    # Expand variations a bit
    def jitter_case(s: str) -> str:
        if random.random() < 0.15:
            return s.upper()
        if random.random() < 0.15:
            return s.lower()
        return s

    wrote = 0
    with out_path.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            u, a = random.choice(pool)
            u = jitter_case(u)
            row = {"instruction": f"User: {norm(u)}\nAssistant:", "output": norm(a)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"Saved: {out_path} | rows={wrote:,}")


if __name__ == "__main__":
    main()
