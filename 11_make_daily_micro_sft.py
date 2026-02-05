#!/usr/bin/env python3
"""
Build a daily micro-retrain SFT JSONL from chat logs + anchor datasets.

Input logs are expected from 09_chat.py turn logging:
  {"ts_utc":"...","mode":"...","user":"...","answer":"..."}

Output rows:
  {"instruction":"User: ...\\nAssistant:","output":"..."}
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional


GENERIC_BAD_ANSWER_HINTS = [
    "how can i help",
    "i'm doing well",
    "i am doing well",
    "happy to help",
    "no problem",
    "jump to content",
    "main menu",
    "markdown content",
    "url source",
    "references:",
]


def norm(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def strip_source_block(answer: str) -> str:
    a = norm(answer)
    # Drop final Sources/Source section used by the chat runtime.
    a = re.sub(r"\n{2,}Sources?:\s*\n(?:- .+\n?)+$", "", a, flags=re.IGNORECASE)
    a = re.sub(r"\n{2,}Source:\s*\n- .+$", "", a, flags=re.IGNORECASE | re.DOTALL)
    return norm(a)


def to_instruction(user_text: str) -> str:
    u = norm(user_text)
    if u.lower().startswith("user:"):
        u = u.split(":", 1)[1].strip()
    return f"User: {u}\nAssistant:"


def looks_good_pair(user_text: str, answer: str) -> bool:
    u = norm(user_text)
    a = norm(answer)
    if not u or not a:
        return False
    if len(a) > 1200:
        return False

    low_a = a.lower()
    if any(h in low_a for h in GENERIC_BAD_ANSWER_HINTS):
        return False
    if "[[" in a:
        return False
    if "http://" in low_a or "https://" in low_a:
        return False
    if a.count("-") > 80:
        return False

    if "answer with a number only" in u.lower():
        if re.fullmatch(r"-?\d+(?:\.\d+)?", a) is None:
            return False

    if "reply yes or no only" in u.lower():
        if a.strip().upper() not in {"YES", "NO"}:
            return False

    if "exactly 2 bullet points" in u.lower():
        bullets = [ln for ln in a.splitlines() if ln.strip().startswith("-")]
        if len(bullets) != 2:
            return False

    return True


def fp_row(row: Dict[str, str]) -> str:
    s = (row["instruction"] + "\n" + row["output"]).encode("utf-8", errors="ignore")
    return hashlib.sha1(s).hexdigest()


def sample_from(pool: List[Dict[str, str]], n: int, rng: random.Random) -> List[Dict[str, str]]:
    if not pool or n <= 0:
        return []
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


def load_anchor(path: str) -> List[Dict[str, str]]:
    rows = read_jsonl(Path(path))
    out: List[Dict[str, str]] = []
    for r in rows:
        ins = norm(str(r.get("instruction", "")))
        outp = norm(str(r.get("output", "")))
        if ins and outp:
            out.append({"instruction": ins, "output": outp})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn_log", default="data/chat_turns_log.jsonl")
    ap.add_argument("--web_log", default="data/web_chat_log.jsonl")
    ap.add_argument(
        "--include_modes",
        default=(
            "url,entity,latest,latest_ddg_fallback,web_general,web_general_news_fallback,web_no_results,"
            "confidence_entity_fallback,confidence_latest_fallback,confidence_web_general_fallback,"
            "confidence_web_no_results,"
            "summary_fallback,math_fallback"
        ),
        help="Comma-separated modes to include from logs",
    )
    ap.add_argument("--include_model", action="store_true", help="Also include mode=model rows")
    ap.add_argument("--keep_sources", action="store_true", help="Keep Source/Sources blocks in outputs")
    ap.add_argument("--max_new_rows", type=int, default=30000)
    ap.add_argument("--out_new", default="data/daily_new_sft.jsonl")

    # Final mixed dataset (new + anchors)
    ap.add_argument("--out", default="data/sft_daily_micro.jsonl")
    ap.add_argument("--max_rows", type=int, default=120000)
    ap.add_argument("--new_ratio", type=float, default=0.25)
    ap.add_argument("--summarize_ratio", type=float, default=0.45)
    ap.add_argument("--routing_ratio", type=float, default=0.15)
    ap.add_argument("--summarize", default="data/summarize_sft.jsonl")
    ap.add_argument("--chat", default="data/basic_chat_sft.jsonl")
    ap.add_argument("--routing", default="data/hard_cases_sft.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    include_modes = {m.strip() for m in args.include_modes.split(",") if m.strip()}
    if args.include_model:
        include_modes.add("model")

    # 1) Build daily-new rows from logs
    log_rows = read_jsonl(Path(args.turn_log))
    # Backfill old sessions where only web_log exists.
    log_rows.extend(read_jsonl(Path(args.web_log)))

    daily_new: List[Dict[str, str]] = []
    skipped_mode = 0
    skipped_bad = 0
    for r in log_rows:
        mode = norm(str(r.get("mode", ""))).lower()
        user = norm(str(r.get("user", "") or r.get("query", "")))
        ans = norm(str(r.get("answer", "")))
        if mode and include_modes and mode not in include_modes:
            skipped_mode += 1
            continue
        if not args.keep_sources:
            ans = strip_source_block(ans)
        if not looks_good_pair(user, ans):
            skipped_bad += 1
            continue
        daily_new.append({"instruction": to_instruction(user), "output": ans})

    # Dedup new rows
    seen_new = set()
    uniq_new: List[Dict[str, str]] = []
    for row in daily_new:
        h = fp_row(row)
        if h in seen_new:
            continue
        seen_new.add(h)
        uniq_new.append(row)
    daily_new = uniq_new

    if args.max_new_rows > 0 and len(daily_new) > args.max_new_rows:
        daily_new = rng.sample(daily_new, args.max_new_rows)

    out_new = Path(args.out_new)
    out_new.parent.mkdir(parents=True, exist_ok=True)
    with out_new.open("w", encoding="utf-8") as f:
        for row in daily_new:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 2) Mix with anchor datasets
    summarize = load_anchor(args.summarize)
    chat = load_anchor(args.chat)
    routing = load_anchor(args.routing)

    max_rows = max(1, int(args.max_rows))
    new_ratio = min(max(args.new_ratio, 0.0), 0.8)
    summarize_ratio = min(max(args.summarize_ratio, 0.0), 0.9)
    routing_ratio = min(max(args.routing_ratio, 0.0), 0.5)

    n_new = int(max_rows * new_ratio)
    n_sum = int(max_rows * summarize_ratio)
    n_routing = int(max_rows * routing_ratio)
    n_chat = max_rows - n_new - n_sum - n_routing
    if n_chat < 0:
        n_chat = 0

    merged: List[Dict[str, str]] = []
    merged.extend(sample_from(daily_new, n_new, rng))
    merged.extend(sample_from(summarize, n_sum, rng))
    merged.extend(sample_from(routing, n_routing, rng))
    merged.extend(sample_from(chat, n_chat, rng))

    # If we are short due to missing pools, top-up from available data.
    pools = [p for p in [daily_new, summarize, routing, chat] if p]
    while len(merged) < max_rows and pools:
        merged.append(rng.choice(rng.choice(pools)))

    rng.shuffle(merged)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Saved new: {out_new} | rows={len(daily_new)} "
        f"(skipped_mode={skipped_mode}, skipped_bad={skipped_bad})"
    )
    print(
        f"Saved mixed: {out} | rows={len(merged)} "
        f"mix_target(new/sum/chat/routing)=({n_new}/{n_sum}/{n_chat}/{n_routing})"
    )


if __name__ == "__main__":
    main()
