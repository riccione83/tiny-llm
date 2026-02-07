#!/usr/bin/env python3
"""
Create a clean, high-quality instruction dataset for instruction fine-tuning.

Sources:
- OpenHermes 2.5 (tasks)
- OpenAssistant (short conversational Q&A)

Output:
- data/instruct_v4.json
"""

from datasets import load_dataset
from pathlib import Path
import json
import random
import re

# -----------------------------
# Config
# -----------------------------
OUT_FILE = Path("data/instruct_v4.json")
OUT_FILE.parent.mkdir(exist_ok=True)

MAX_HERMES = 80_000
MAX_CHAT   = 60_000
SEED = 42

MIN_INSTR = 8
MAX_INSTR = 600
MIN_OUT   = 2
MAX_OUT   = 800

random.seed(SEED)

# -----------------------------
# Helpers
# -----------------------------
_space = re.compile(r"[ \t]+")
_many_blank_lines = re.compile(r"\n{3,}")

def normalize_preserve_newlines(text: str) -> str:
    """
    Keep newlines (structure), but normalize spaces within lines.
    Also trim excessive blank lines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [_space.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _many_blank_lines.sub("\n\n", text)
    return text.strip()

def looks_english(text: str) -> bool:
    # English-dominant heuristic (less strict than 0.92 to keep normal chat)
    ascii_ratio = sum(ord(c) < 128 for c in text) / max(len(text), 1)
    return ascii_ratio > 0.88

def is_code_heavy(text: str) -> bool:
    t = text.strip()
    if "```" in t:
        return True
    # lots of typical code tokens
    code_hits = sum(tok in t for tok in ["import ", "def ", "class ", "{", "}", ";", "=>", "</", "/>", "SELECT ", "INSERT "])
    if code_hits >= 2:
        return True
    # many non-letter symbols (rough heuristic)
    sym = sum(not c.isalnum() and not c.isspace() for c in t)
    if len(t) > 0 and (sym / len(t)) > 0.18:
        return True
    return False

def has_reasonable_end(text: str) -> bool:
    # Avoid many incomplete assistant fragments
    return len(text) < 40 or text.rstrip()[-1] in ".!?\"'`)]}"

_BAD_DISCLAIMERS = (
    "as an ai language model",
    "as a language model",
    "i am an ai language model",
    "i'm an ai language model",
    "i am a language model",
    "i'm a language model",
    "i cannot provide",
    "i can't provide",
    "i cannot help",
    "i can't help",
    "i am unable",
    "i'm unable",
    "i do not have the ability",
    "i don't have the ability",
    "i don't have access",
    "i do not have access",
)

def is_disclaimer(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _BAD_DISCLAIMERS)

def is_low_quality_out(out: str) -> bool:
    t = out.lower()
    if is_disclaimer(out):
        return True
    # Avoid clear multi-turn leakage
    if "user:" in t or "assistant:" in t:
        return True
    # Avoid super repetitive outputs
    words = out.split()
    if len(words) >= 20 and len(set(words)) < max(10, len(words) // 6):
        return True
    return False

def valid_pair(instr: str, out: str) -> bool:
    return (
        MIN_INSTR <= len(instr) <= MAX_INSTR and
        MIN_OUT   <= len(out)   <= MAX_OUT and
        looks_english(instr) and
        looks_english(out)
    )

# -----------------------------
# Load OpenHermes (instructions)
# -----------------------------
print("Loading OpenHermes 2.5...")
hermes = load_dataset("teknium/OpenHermes-2.5", split="train")

data = []
for row in hermes:
    conv = row.get("conversations")
    if not conv or len(conv) < 2:
        continue

    instr = normalize_preserve_newlines(conv[0].get("value", ""))
    out   = normalize_preserve_newlines(conv[1].get("value", ""))

    # Skip code-heavy tasks for now (chat-first goal)
    if is_code_heavy(out) or is_code_heavy(instr):
        continue
    if is_low_quality_out(out):
        continue
    if valid_pair(instr, out):
        data.append({"instruction": instr, "output": out})
    if len(data) >= MAX_HERMES:
        break

print(f"OpenHermes kept: {len(data)}")

# -----------------------------
# Load OpenAssistant (chat → Q&A)
# -----------------------------
print("Loading OpenAssistant...")
oa = load_dataset("OpenAssistant/oasst1", split="train")

by_id = {row["message_id"]: row for row in oa}
chat_added = 0

for row in oa:
    if row.get("role") != "assistant":
        continue

    parent = row.get("parent_id")
    if not parent or parent not in by_id:
        continue
    if by_id[parent].get("role") != "prompter":
        continue

    user = normalize_preserve_newlines(by_id[parent].get("text", ""))
    assistant = normalize_preserve_newlines(row.get("text", ""))

    # Keep chat short & crisp
    if not (3 <= len(user) <= 200 and 3 <= len(assistant) <= 400):
        continue
    if not (looks_english(user) and looks_english(assistant)):
        continue
    if not has_reasonable_end(assistant):
        continue
    if is_low_quality_out(assistant):
        continue

    data.append({"instruction": user, "output": assistant})
    chat_added += 1
    if chat_added >= MAX_CHAT:
        break

print(f"OpenAssistant kept: {chat_added}")

# -----------------------------
# Shuffle & save
# -----------------------------
random.shuffle(data)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print(f"\nSaved {len(data)} examples -> {OUT_FILE}")
