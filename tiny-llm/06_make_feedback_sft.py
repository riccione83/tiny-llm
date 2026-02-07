#!/usr/bin/env python3
"""
Build a large feedback dataset (instruction/chosen) from:
- OpenAssistant/oasst1 (leaf assistant replies)
- databricks/databricks-dolly-15k
- deterministic synthetic templates

Output: feedback/feedback_sft.jsonl (backup existing file)
"""
import json
import os
import random
import re
from pathlib import Path
from datetime import datetime

from datasets import load_dataset

OUT_PATH = Path("feedback/feedback_sft.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_TOTAL = 100_000
TARGET_OASST = 40_000
TARGET_DOLLY = 15_000

MIN_INSTR = 3
MAX_INSTR = 400
MIN_OUT = 2
MAX_OUT = 600

random.seed(SEED)

_space = re.compile(r"[ \t]+")
_many_blank = re.compile(r"\n{3,}")

BAD_DISCLAIMERS = (
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

CODE_TOKENS = ["import ", "def ", "class ", "{", "}", ";", "=>", "</", "/>", "SELECT ", "INSERT "]


def normalize_preserve_newlines(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [_space.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _many_blank.sub("\n\n", text)
    return text.strip()


def looks_english(text: str) -> bool:
    ascii_ratio = sum(ord(c) < 128 for c in text) / max(len(text), 1)
    return ascii_ratio > 0.88


def is_code_heavy(text: str) -> bool:
    t = text.strip()
    if "```" in t:
        return True
    code_hits = sum(tok in t for tok in CODE_TOKENS)
    if code_hits >= 2:
        return True
    sym = sum(not c.isalnum() and not c.isspace() for c in t)
    if len(t) > 0 and (sym / len(t)) > 0.18:
        return True
    return False


def is_disclaimer(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in BAD_DISCLAIMERS)


def has_reasonable_end(text: str) -> bool:
    return len(text) < 40 or text.rstrip()[-1] in ".!?\"'`)]}"


def valid_pair(instr: str, out: str) -> bool:
    if not (MIN_INSTR <= len(instr) <= MAX_INSTR):
        return False
    if not (MIN_OUT <= len(out) <= MAX_OUT):
        return False
    if not looks_english(instr) or not looks_english(out):
        return False
    if is_disclaimer(out):
        return False
    if "user:" in out.lower() or "assistant:" in out.lower():
        return False
    if is_code_heavy(instr) or is_code_heavy(out):
        return False
    if not has_reasonable_end(out):
        return False
    return True


def load_oasst_pairs(limit: int):
    print("Loading OpenAssistant/oasst1...")
    oa = load_dataset("OpenAssistant/oasst1", split="train")
    by_id = {row["message_id"]: row for row in oa}

    children_ids = {}
    for row in oa:
        pid = row.get("parent_id")
        if pid:
            children_ids.setdefault(pid, []).append(row["message_id"])

    pairs = []
    for row in oa:
        if row.get("role") != "assistant":
            continue
        pid = row.get("parent_id")
        if not pid or pid not in by_id:
            continue
        if by_id[pid].get("role") != "prompter":
            continue
        if row.get("message_id") in children_ids:
            continue

        instr = normalize_preserve_newlines(by_id[pid].get("text", ""))
        out = normalize_preserve_newlines(row.get("text", ""))
        if not instr or not out:
            continue
        if valid_pair(instr, out):
            pairs.append((instr, out))
            if len(pairs) >= limit:
                break

    print(f"OpenAssistant kept: {len(pairs)}")
    return pairs


def load_dolly_pairs(limit: int):
    print("Loading databricks/databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    pairs = []
    for row in ds:
        instr = normalize_preserve_newlines(row.get("instruction", ""))
        context = normalize_preserve_newlines(row.get("context", ""))
        if context:
            instr = f"{instr}\n\nContext:\n{context}"
        out = normalize_preserve_newlines(row.get("response", ""))
        if not instr or not out:
            continue
        if valid_pair(instr, out):
            pairs.append((instr, out))
            if len(pairs) >= limit:
                break
    print(f"Dolly kept: {len(pairs)}")
    return pairs


def build_synthetic_pairs(target: int):
    prompts = []

    chat_prompts = [
        ("Hi!", "Hi there! How can I help you today?"),
        ("Hello!", "Hello! What would you like to talk about?"),
        ("How are you?", "I'm doing well, thanks for asking! How are you?"),
        ("What's up?", "Not much, just here to help. What's up with you?"),
        ("Good morning", "Good morning! Anything I can help with?"),
        ("Good evening", "Good evening! How can I help?"),
        ("Thanks", "You're welcome!"),
        ("Thank you", "You're welcome!"),
        ("Bye", "Bye! Have a great day."),
        ("I feel stressed. Any quick tips?", "Try box breathing: 4 in, hold 4, out 4, hold 4."),
        ("I'm anxious. Any quick tips?", "Try grounding: name 5 things you see, 4 you feel, 3 you hear."),
        ("Can you motivate me today?", "Pick one tiny task and do it for 5 minutes. Momentum will follow."),
        ("Any tips to relax quickly?", "Try a slow 4-4-4-4 breath and loosen your shoulders."),
        ("Say something encouraging.", "You've got this. Start small and keep going."),
        ("How can I focus better?", "Remove distractions and do 25 minutes of focused work, then a 5-minute break."),
        ("What should I do after work to unwind?", "A short walk, a shower, or a light hobby works well."),
        ("Tell me a short joke.", "Why did the scarecrow win an award? Because he was outstanding in his field."),
    ]
    prompts.extend(chat_prompts)

    defs = {
        "What is a transformer model?": "A transformer is a neural network that uses attention to process sequences efficiently.",
        "Explain gradient descent simply.": "Gradient descent minimizes loss by taking small steps in the direction that reduces error.",
        "What is overfitting?": "Overfitting is when a model learns training data too well and performs poorly on new data.",
        "What is attention in ML?": "Attention lets a model focus on the most relevant parts of the input.",
        "What is tokenization?": "Tokenization splits text into smaller units (tokens) for a model to process.",
        "What is a learning rate?": "The learning rate controls how big each parameter update step is during training.",
        "What is dropout?": "Dropout randomly zeroes activations during training to reduce overfitting.",
        "What is a validation set?": "A validation set is held-out data used to tune and monitor model performance.",
    }
    prompts.extend(list(defs.items()))

    for a in range(2, 302):
        b = a + 1
        prompts.append((f"What is {a}+{b}?", f"{a}+{b} = {a + b}."))
    for a in range(10, 310, 3):
        b = 7
        prompts.append((f"What is {a}-{b}?", f"{a}-{b} = {a - b}."))
    for a in range(3, 203):
        b = 4
        prompts.append((f"What is {a}*{b}?", f"{a}*{b} = {a * b}."))
    for a in range(20, 620, 5):
        b = 5
        prompts.append((f"What is {a}/{b}?", f"{a}/{b} = {a // b}."))

    pairs = []
    i = 0
    while len(pairs) < target:
        instr, out = prompts[i % len(prompts)]
        instr = normalize_preserve_newlines(instr)
        out = normalize_preserve_newlines(out)
        if valid_pair(instr, out):
            pairs.append((instr, out))
        i += 1

    random.shuffle(pairs)
    return pairs[:target]


def main():
    pairs = []

    pairs.extend(load_oasst_pairs(TARGET_OASST))
    pairs.extend(load_dolly_pairs(TARGET_DOLLY))

    remaining = max(0, TARGET_TOTAL - len(pairs))
    print(f"Synthetic target: {remaining}")
    pairs.extend(build_synthetic_pairs(remaining))

    random.shuffle(pairs)

    if OUT_PATH.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = OUT_PATH.with_name(f"feedback_sft_prev_{ts}.jsonl")
        OUT_PATH.replace(backup)
        print(f"Backed up existing feedback file -> {backup}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for instr, out in pairs:
            f.write(json.dumps({"instruction": instr, "chosen": out}, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs):,} examples -> {OUT_PATH}")


if __name__ == "__main__":
    main()
