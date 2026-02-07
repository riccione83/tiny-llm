#!/usr/bin/env python3
"""
01_make_chat_corpus_and_tokenize.py

Chat-first corpus creation + pretokenization (Windows-safe, low RAM)

Outputs:
- data/chat_corpus_v1.txt
- data/chat_corpus_v1_tokens.npy  (memmapped .npy of token ids)

Fixes (IMPORTANT):
- Select BEST assistant reply per prompter (rank/score) to avoid bad pairings
- ENGLISH-only via row["lang"] when available (+ extra mojibake filter)
- Tokenize by SAMPLE blocks (not line-by-line) with one EOS per sample
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
SEED = 42

OA_DATASET = "OpenAssistant/oasst1"
OA_SPLIT = "train"

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)
OUT_TEXT = OUT_DIR / "chat_corpus_v1.txt"
OUT_TOKENS = OUT_DIR / "chat_corpus_v1_tokens.npy"

TOKENIZER_PATH = "llm_tokenizer.model"

MAX_OASST_PAIRS = 40_000
MAX_BST_PAIRS = 60_000
MAX_HERMES_PAIRS = 60_000

MIN_USER_CHARS = 2
MAX_USER_CHARS = 600
MIN_ASSIST_CHARS = 2
MAX_ASSIST_CHARS = 900

LINES_PER_CHUNK = 6000
TEMP_DIR = OUT_DIR / "tmp_tok_chunks"

DTYPE = np.uint32

# Small, high-quality chatty seed pairs to improve basic friendliness
EXTRA_CHATTY_PAIRS = [
    ("Hi", "Hi! How can I help you today?"),
    ("Hello", "Hello! What would you like to talk about?"),
    ("Hey", "Hey! How's your day going?"),
    ("How are you?", "I'm doing well—thanks for asking! How are you?"),
    ("Good morning", "Good morning! Anything I can help with?"),
    ("Good evening", "Good evening! How can I help?"),
    ("What's up?", "Not much—I'm here to help. What's up with you?"),
    ("What is 2+2?", "2+2 = 4."),
    ("What is 3+5?", "3+5 = 8."),
    ("What is 10-4?", "10-4 = 6."),
    ("What is 6*7?", "6*7 = 42."),
    ("What is 20/5?", "20/5 = 4."),
    ("Thanks", "You're welcome!"),
    ("Thank you", "You're welcome!"),
    ("Bye", "Bye! Have a great day."),
]

# -----------------------------
# TEXT NORMALIZATION (PRESERVE NEWLINES)
# -----------------------------
_space = re.compile(r"[ \t]+")
_many_blank_lines = re.compile(r"\n{3,}")

def normalize_preserve_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [_space.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _many_blank_lines.sub("\n\n", text)
    return text.strip()

def collapse_blank_lines(text: str) -> str:
    # Ensure no empty lines inside a sample (blank lines delimit samples).
    return re.sub(r"\n{2,}", "\n", text)

def normalize_user_line(text: str) -> str:
    # Keep user prompts single-line for consistent "User:" / "Assistant:" framing
    u = re.sub(r"\s+", " ", text).strip()
    if u.startswith("User: "):
        u = u[len("User: "):]
    return u

def looks_english(text: str) -> bool:
    """
    Relaxed heuristic: allow normal unicode punctuation.
    Only reject if it's heavily non-ascii.
    """
    ascii_ratio = sum(ord(c) < 128 for c in text) / max(1, len(text))
    return ascii_ratio > 0.88

def has_mojibake(text: str) -> bool:
    # Common UTF-8 decoding artifacts in scraped datasets
    return ("Ã" in text) or ("�" in text)


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

def is_low_quality_assistant(text: str) -> bool:
    t = text.lower()
    if any(b in t for b in _BAD_DISCLAIMERS):
        return True
    if "user:" in t or "assistant:" in t:
        return True
    words = text.split()
    if len(words) >= 30 and len(set(words)) < max(10, len(words) // 6):
        return True
    return False

def row_lang_is_en(row: dict) -> Optional[bool]:
    """
    Returns:
      True/False if 'lang' exists,
      None if not present.
    """
    lang = row.get("lang")
    if not lang:
        return None
    return str(lang).lower().startswith("en")

def has_topic_overlap(user: str, assistant: str) -> bool:
    import re
    u = set(re.findall(r"[a-z]{4,}", user.lower()))
    a = set(re.findall(r"[a-z]{4,}", assistant.lower()))
    if len(u) < 3:
        return True
    overlap = u & a
    return len(overlap) >= max(1, len(u) // 6)

def build_bst_pairs(max_pairs: int) -> List[Tuple[str, str]]:
    """
    Pull short, chatty pairs from BlendedSkillTalk.
    """
    try:
        bst = load_dataset("blended_skill_talk", split="train")
    except Exception as e:
        print(f"BST load failed: {e}")
        return []

    pairs: List[Tuple[str, str]] = []
    for ex in bst:
        msgs = ex.get("free_messages") or []
        if len(msgs) < 2:
            continue
        # pair up consecutive user/assistant turns
        for i in range(0, len(msgs) - 1, 2):
            u = normalize_preserve_newlines(msgs[i])
            a = normalize_preserve_newlines(msgs[i + 1])
            if not u or not a:
                continue
            if not (MIN_USER_CHARS <= len(u) <= MAX_USER_CHARS and MIN_ASSIST_CHARS <= len(a) <= MAX_ASSIST_CHARS):
                continue
            if not (looks_english(u) and looks_english(a)):
                continue
            if is_low_quality_assistant(a):
                continue
            if is_code_heavy(u) or is_code_heavy(a):
                continue
            if not has_reasonable_end(a):
                continue
            pairs.append((u, a))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs

# -----------------------------
# BUILD CHAT PAIRS FROM OASST1
# -----------------------------
def build_chat_pairs_from_oasst1(max_pairs: int) -> List[Tuple[str, str]]:
    print(f"Loading {OA_DATASET} ({OA_SPLIT})...")

    oa = load_dataset(OA_DATASET, split=OA_SPLIT)

    # Index rows by id
    by_id: Dict[str, dict] = {row["message_id"]: row for row in oa}

    # Build reverse index: parent_id -> list of children ids
    children_ids: Dict[str, List[str]] = {}
    for row in oa:
        pid = row.get("parent_id")
        if pid:
            children_ids.setdefault(pid, []).append(row["message_id"])

    # Group assistant children by parent_id
    children: Dict[str, List[dict]] = {}
    assistants_seen = 0
    no_parent = 0
    parent_missing = 0
    parent_not_prompter = 0

    for row in oa:
        if row.get("role") != "assistant":
            continue
        assistants_seen += 1

        pid = row.get("parent_id")
        if not pid:
            no_parent += 1
            continue
        if pid not in by_id:
            parent_missing += 1
            continue
        if by_id[pid].get("role") != "prompter":
            parent_not_prompter += 1
            continue

        children.setdefault(pid, []).append(row)

    # Helper to pick best assistant for a parent (rank/score if present)
    def choose_best_assistant(rows: List[dict]) -> dict:
        # Lower rank is better if present; otherwise higher score if present
        def key(r: dict):
            rank = r.get("rank")
            score = r.get("score")
            # rank: smaller is better; score: larger is better
            rank_key = int(rank) if isinstance(rank, (int, np.integer)) else 10_000
            score_key = float(score) if isinstance(score, (float, int, np.floating, np.integer)) else -1e9
            return (rank_key, -score_key)  # prefer low rank, then high score
        rows = sorted(rows, key=key)
        return rows[0]

    pairs: List[Tuple[str, str]] = []

    # Debug counters (OASST only)
    filtered_len = 0
    filtered_lang = 0
    filtered_quality = 0
    filtered_mojibake = 0
    filtered_topic = 0
    filtered_code = 0
    filtered_end = 0

    # Iterate over parents; choose best assistant child
    for parent_id, kids in tqdm(children.items(), desc="Pairing (best reply per prompt)"):
        if len(pairs) >= max_pairs:
            break

        parent_row = by_id[parent_id]
        # keep only assistant replies that are LEAVES (no children)
        leaf_kids = [
            r for r in kids
            if r.get("message_id") not in children_ids
        ]

        if not leaf_kids:
            continue

        best = choose_best_assistant(leaf_kids)

        user = normalize_preserve_newlines(parent_row.get("text", ""))
        assistant = normalize_preserve_newlines(best.get("text", ""))

        if has_mojibake(user) or has_mojibake(assistant):
            filtered_mojibake += 1
            continue

        # Strong language filter if available
        pl = row_lang_is_en(parent_row)
        al = row_lang_is_en(best)
        if pl is not None and not pl:
            filtered_lang += 1
            continue
        if al is not None and not al:
            filtered_lang += 1
            continue

        # Fallback heuristic if lang missing
        if (pl is None or al is None) and not (looks_english(user) and looks_english(assistant)):
            filtered_lang += 1
            continue

        if not (MIN_USER_CHARS <= len(user) <= MAX_USER_CHARS and MIN_ASSIST_CHARS <= len(assistant) <= MAX_ASSIST_CHARS):
            filtered_len += 1
            continue

        if is_low_quality_assistant(assistant):
            filtered_quality += 1
            continue

        if is_code_heavy(user) or is_code_heavy(assistant):
            filtered_code += 1
            continue

        if not has_reasonable_end(assistant):
            filtered_end += 1
            continue

        if not has_topic_overlap(user, assistant):
            filtered_topic += 1
            continue

        pairs.append((user, assistant))

    print(f"Kept chat pairs: {len(pairs):,}")
    print(
        "Debug:"
        f" assistants_seen={assistants_seen:,}"
        f" no_parent={no_parent:,}"
        f" parent_missing={parent_missing:,}"
        f" parent_not_prompter={parent_not_prompter:,}"
        f" filtered_len={filtered_len:,}"
        f" filtered_lang={filtered_lang:,}"
        f" filtered_quality={filtered_quality:,}"
        f" filtered_mojibake={filtered_mojibake:,}"
        f" filtered_topic={filtered_topic:,}"
        f" filtered_code={filtered_code:,}"
        f" filtered_end={filtered_end:,}"
        f" parents_with_assistant_children={len(children):,}"
    )
    if EXTRA_CHATTY_PAIRS:
        # Append tiny curated chatty set for basic greeting/math coverage
        for u, a in EXTRA_CHATTY_PAIRS:
            pairs.append((u, a))
    return pairs

def build_openhermes_pairs(max_pairs: int) -> List[Tuple[str, str]]:
    """
    Use OpenHermes as short instruction->response pairs (chatty constraints).
    """
    try:
        hermes = load_dataset("teknium/OpenHermes-2.5", split="train")
    except Exception as e:
        print(f"OpenHermes load failed: {e}")
        return []

    pairs: List[Tuple[str, str]] = []
    for row in hermes:
        conv = row.get("conversations")
        if not conv or len(conv) < 2:
            continue
        u = normalize_preserve_newlines(conv[0].get("value", ""))
        a = normalize_preserve_newlines(conv[1].get("value", ""))
        if not u or not a:
            continue
        if not (MIN_USER_CHARS <= len(u) <= 300 and MIN_ASSIST_CHARS <= len(a) <= 600):
            continue
        if not (looks_english(u) and looks_english(a)):
            continue
        if is_low_quality_assistant(a):
            continue
        if is_code_heavy(u) or is_code_heavy(a):
            continue
        if not has_reasonable_end(a):
            continue
        if not has_topic_overlap(u, a):
            continue
        pairs.append((u, a))
        if len(pairs) >= max_pairs:
            break
    return pairs

def write_chat_corpus(pairs: List[Tuple[str, str]], out_path: Path) -> None:
    print(f"Writing corpus -> {out_path}")
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for (u, a) in pairs:
            u = collapse_blank_lines(u)
            a = collapse_blank_lines(a)
            u = normalize_user_line(u)
            f.write("User: " + u + "\n")
            f.write("Assistant: " + a + "\n")
            f.write("\n")
    print("Corpus written")

# -----------------------------
# TOKENIZATION (LOW RAM, EOS PER SAMPLE)
# -----------------------------
def chunked_samples(path: Path, samples_per_chunk: int):
    """
    Reads OUT_TEXT where samples are separated by blank lines.
    Yields a list[str] of samples (each sample is a multi-line string).
    """
    buf_lines: List[str] = []
    chunk: List[str] = []

    def flush_sample():
        nonlocal buf_lines, chunk
        if buf_lines:
            sample = "\n".join(buf_lines).strip()
            if sample:
                chunk.append(sample)
            buf_lines = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                flush_sample()
                if len(chunk) >= samples_per_chunk:
                    yield chunk
                    chunk = []
            else:
                buf_lines.append(line.rstrip("\n"))

    flush_sample()
    if chunk:
        yield chunk

def tokenize_samples(samples: List[str], sp: spm.SentencePieceProcessor, eos_id: Optional[int]) -> np.ndarray:
    toks: List[int] = []
    for s in samples:
        toks.extend(sp.encode(s, out_type=int))
        if eos_id is not None:
            toks.append(eos_id)  # ONE EOS per sample
    return np.asarray(toks, dtype=DTYPE)

def pretokenize_to_npy(text_path: Path, out_npy: Path, tokenizer_path: str) -> None:
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    vocab = sp.get_piece_size()
    eos_id = sp.eos_id()
    print(f"Tokenizer loaded | vocab={vocab:,} | eos_id={eos_id} | dtype={DTYPE.__name__}")

    TEMP_DIR.mkdir(exist_ok=True)

    if out_npy.exists():
        print(f"Removing existing: {out_npy}")
        out_npy.unlink()

    chunk_paths: List[Path] = []
    print("Tokenizing into temporary chunks...")
    for i, samples in enumerate(tqdm(chunked_samples(text_path, samples_per_chunk=2000), desc="Tokenizing")):
        arr = tokenize_samples(samples, sp, eos_id)
        p = TEMP_DIR / f"chunk_{i:06d}.npy"
        np.save(p, arr)
        chunk_paths.append(p)

    total_len = 0
    for p in tqdm(chunk_paths, desc="Measuring chunks"):
        a = np.load(p, mmap_mode="r")
        total_len += int(len(a))

    print(f"Total tokens: {total_len:,}")

    print(f"Writing final memmap -> {out_npy}")
    final = np.lib.format.open_memmap(out_npy, mode="w+", dtype=DTYPE, shape=(total_len,))

    offset = 0
    for p in tqdm(chunk_paths, desc="Writing"):
        a = np.load(p, mmap_mode="r")
        n = int(len(a))
        final[offset:offset+n] = a
        offset += n

    del final

    print(f"Pretokenized saved: {out_npy} ({out_npy.stat().st_size / 1e9:.2f} GB)")

    print("Cleaning up temp chunks...")
    for p in chunk_paths:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        TEMP_DIR.rmdir()
    except OSError:
        pass

def main():
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    pairs: List[Tuple[str, str]] = []

    oasst_pairs = build_chat_pairs_from_oasst1(MAX_OASST_PAIRS)
    if oasst_pairs:
        print(f"OASST pairs added: {len(oasst_pairs):,}")
        pairs.extend(oasst_pairs)

    bst_pairs = build_bst_pairs(MAX_BST_PAIRS)
    if bst_pairs:
        print(f"BST pairs added: {len(bst_pairs):,}")
        pairs.extend(bst_pairs)

    hermes_pairs = build_openhermes_pairs(MAX_HERMES_PAIRS)
    if hermes_pairs:
        print(f"OpenHermes pairs added: {len(hermes_pairs):,}")
        pairs.extend(hermes_pairs)

    if EXTRA_CHATTY_PAIRS:
        pairs.extend(EXTRA_CHATTY_PAIRS)

    # final dedupe
    seen = set()
    deduped = []
    for u, a in pairs:
        key = (u, a)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((u, a))
    pairs = deduped
    if not pairs:
        raise RuntimeError("No pairs were kept. Check the debug counters printed above.")

    write_chat_corpus(pairs, OUT_TEXT)
    pretokenize_to_npy(OUT_TEXT, OUT_TOKENS, TOKENIZER_PATH)

    print("\nDone.")
    print(f"Corpus:      {OUT_TEXT}")
    print(f"Pretokens:   {OUT_TOKENS}")

if __name__ == "__main__":
    main()
