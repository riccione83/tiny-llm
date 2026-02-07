#!/usr/bin/env python3
"""
05_create_instruct_v2.py

Create a large, clean instruction/chat dataset for base training.

Sources:
- OpenHermes 2.5 (instruction/task pairs)
- OpenAssistant (leaf-only, prompter->assistant)
- BlendedSkillTalk (chatty)

Output:
- data/instruct_v2.jsonl
"""

from datasets import load_dataset
from pathlib import Path
import json
import random
import re

# -----------------------------
# Config
# -----------------------------
OUT_FILE = Path("data/instruct_v2.jsonl")
OUT_FILE.parent.mkdir(exist_ok=True)

MAX_HERMES = 200_000
MAX_OASST  = 120_000
MAX_BST    = 120_000
MAX_TS_SAMPLES = None  # None = size-based stop
TARGET_TS_GB = 40
SEED = 42

MIN_INSTR = 3
MAX_INSTR = 600
MIN_OUT   = 2
MAX_OUT   = 800
MAX_CODE_CHARS = 8000

random.seed(SEED)

# -----------------------------
# Helpers
# -----------------------------
_space = re.compile(r"[ \t]+")
_many_blank_lines = re.compile(r"\n{3,}")

def normalize_preserve_newlines(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [_space.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _many_blank_lines.sub("\n\n", text)
    return text.strip()

def looks_english(text: str) -> bool:
    ascii_ratio = sum(ord(c) < 128 for c in text) / max(len(text), 1)
    return ascii_ratio > 0.88

def is_code_heavy(text: str) -> bool:
    t = text.strip()
    if "```" in t:
        return True
    code_hits = sum(tok in t for tok in ["import ", "def ", "class ", "{", "}", ";", "=>", "</", "/>", "SELECT ", "INSERT "])
    if code_hits >= 2:
        return True
    sym = sum(not c.isalnum() and not c.isspace() for c in t)
    if len(t) > 0 and (sym / len(t)) > 0.18:
        return True
    return False

def has_reasonable_end(text: str) -> bool:
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
    "i'm just an ai",
    "i am just an ai",
)

def is_disclaimer(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _BAD_DISCLAIMERS)

def is_low_quality_out(out: str) -> bool:
    t = out.lower()
    if is_disclaimer(out):
        return True
    if "user:" in t or "assistant:" in t:
        return True
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
# TypeScript code (non-gated: codeparrot/github-code)
# -----------------------------
def iter_typescript_github_code(target_gb: int):
    """
    Stream TypeScript files from codeparrot/github-code (non-gated).
    Stops when written size reaches target_gb.
    """
    try:
        ds = load_dataset("codeparrot/github-code", split="train", streaming=True)
    except Exception as e:
        print(f"GitHub code dataset load failed: {e}")
        return

    bytes_target = int(target_gb * (1024**3))
    bytes_written = 0
    kept = 0

    for row in ds:
        lang = (row.get("language") or "").lower()
        path = row.get("path") or ""
        if lang != "typescript" and not (path.endswith(".ts") or path.endswith(".tsx")):
            continue

        content = row.get("code") or ""
        content = normalize_preserve_newlines(content)
        if not content:
            continue
        if len(content) < 200 or len(content) > MAX_CODE_CHARS:
            continue

        instr = f"Write a TypeScript file based on this path: {path or 'unknown.ts'}"
        out = content

        item = {"instruction": instr, "output": out}
        line = json.dumps(item, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))
        yield line
        bytes_written += line_bytes
        kept += 1
        if bytes_written >= bytes_target:
            print(f"TypeScript target reached: {bytes_written / (1024**3):.2f} GB, samples={kept:,}")
            break

# -----------------------------
# OpenHermes
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
    if is_code_heavy(instr) or is_code_heavy(out):
        continue
    if is_low_quality_out(out):
        continue
    if valid_pair(instr, out):
        data.append({"instruction": instr, "output": out})
    if len(data) >= MAX_HERMES:
        break

print(f"OpenHermes kept: {len(data)}")

# -----------------------------
# OpenAssistant (leaf-only)
# -----------------------------
print("Loading OpenAssistant...")
oa = load_dataset("OpenAssistant/oasst1", split="train")
by_id = {row["message_id"]: row for row in oa}

# Build reverse index for leaf filtering
children_ids = {}
for row in oa:
    pid = row.get("parent_id")
    if pid:
        children_ids.setdefault(pid, []).append(row["message_id"])

chat_added = 0
for row in oa:
    if row.get("role") != "assistant":
        continue
    if row.get("message_id") in children_ids:
        continue
    parent = row.get("parent_id")
    if not parent or parent not in by_id:
        continue
    if by_id[parent].get("role") != "prompter":
        continue

    user = normalize_preserve_newlines(by_id[parent].get("text", ""))
    assistant = normalize_preserve_newlines(row.get("text", ""))

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
    if chat_added >= MAX_OASST:
        break

print(f"OpenAssistant kept: {chat_added}")

# -----------------------------
# BlendedSkillTalk
# -----------------------------
print("Loading BlendedSkillTalk...")
try:
    bst = load_dataset("blended_skill_talk", split="train")
    bst_added = 0
    for ex in bst:
        msgs = ex.get("free_messages") or []
        if len(msgs) < 2:
            continue
        for i in range(0, len(msgs) - 1, 2):
            user = normalize_preserve_newlines(msgs[i])
            assistant = normalize_preserve_newlines(msgs[i + 1])
            if not user or not assistant:
                continue
            if not (looks_english(user) and looks_english(assistant)):
                continue
            if is_low_quality_out(assistant):
                continue
            if not has_reasonable_end(assistant):
                continue
            data.append({"instruction": user, "output": assistant})
            bst_added += 1
            if bst_added >= MAX_BST:
                break
        if bst_added >= MAX_BST:
            break
    print(f"BlendedSkillTalk kept: {bst_added}")
except Exception as e:
    print(f"BST load failed: {e}")

# -----------------------------
# Shuffle & save (JSONL)
# -----------------------------
random.shuffle(data)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for row in data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Append TypeScript code (streamed from codeparrot/github-code)
    print(f"Appending TypeScript code until ~{TARGET_TS_GB} GB...")
    for line in iter_typescript_github_code(TARGET_TS_GB):
        f.write(line)

print(f"\nSaved {len(data)} examples -> {OUT_FILE}")
