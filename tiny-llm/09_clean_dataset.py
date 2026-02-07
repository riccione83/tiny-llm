#!/usr/bin/env python3
import json, re
from pathlib import Path

INP = Path("feedback/feedback_sft.jsonl")
OUT = Path("feedback/feedback_sft_clean.jsonl")

BAD_PREFIXES = (
    "sorry", "as an ai", "i'm just", "i am just", "i cannot", "i can't"
)

def fix_mojibake(s: str) -> str:
    # common broken apostrophes/quotes seen in datasets
    s = s.replace("ÔÇÖ", "'").replace("ÔÇ£", '"').replace("ÔÇ¥", '"')
    s = s.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    s = s.replace("Ã©", "é").replace("Ã¨", "è").replace("Ã¬", "ì").replace("Ã²", "ò").replace("Ã¹", "ù")
    return s

def normalize(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = fix_mojibake(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def is_bad(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return True
    if "�" in t or "Ã" in t or "ÔÇ" in t:
        return True
    # avoid meta/apology style in canonical answers
    if any(t.startswith(p) for p in BAD_PREFIXES):
        return True
    return False

def main():
    if not INP.exists():
        raise FileNotFoundError(INP)

    # dedupe by instruction (keep last occurrence)
    best = {}
    total = 0
    kept = 0
    dropped = 0

    with INP.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            instr = normalize(row.get("instruction", ""))
            chosen = normalize(row.get("chosen", ""))

            if len(instr) < 2 or len(chosen) < 2:
                dropped += 1
                continue
            if is_bad(chosen):
                dropped += 1
                continue
            if len(instr) > 400 or len(chosen) > 800:
                # keep feedback concise
                dropped += 1
                continue

            key = instr.lower()
            best[key] = {"instruction": instr, "chosen": chosen}
            kept += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in best.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"read={total:,} kept_last={len(best):,} dropped={dropped:,}")
    print(f"written -> {OUT}")

if __name__ == "__main__":
    main()
