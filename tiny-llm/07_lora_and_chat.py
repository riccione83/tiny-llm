#!/usr/bin/env python3
"""
07_lora_and_chat.py

FastGPT (184M) — LoRA instruction tuning + interactive chat CLI.

Modes:
  1) LoRA fine-tune:
     python 07_lora_and_chat.py --mode lora

  2) Chat:
     python 07_lora_and_chat.py --mode chat

     python .\07_lora_and_chat.py --mode chat --use_lora

     python .\07_lora_and_chat.py --mode feedback_lora

     Enter → skip (no logging)

u → 👍 good answer (optional to log)

e → ✍️ edit / paste a better answer
→ this becomes a training example

q → quit


Assumes:
- tokenizer: llm_tokenizer.model
- base checkpoint: checkpoints_chat_v1/interrupted_step2032.pt
- instruct data: data/instruct_v4.json

Outputs:
- finetuning/lora_adapter.pt
- finetuning/lora_full_state.pt  (base+LoRA state_dict for convenience)
"""

import os
import math
import json
import argparse
import re
import time
import hashlib
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import sentencepiece as spm
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# ──────────────────────────────
# CUDA speed-ups (safe)
# ──────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ──────────────────────────────
# CONFIG — Model
# ──────────────────────────────
BLOCK_SIZE  = 768
EMBED_DIM   = 896
NUM_HEADS   = 14
NUM_LAYERS  = 16
DROPOUT     = 0.1

# ──────────────────────────────
# PATHS
# ──────────────────────────────
TOKENIZER_PATH = "llm_tokenizer.model"
INSTRUCT_JSON  = "data/instruct_v4.json"

def latest_ckpt(dir_path: str):
    if not os.path.exists(dir_path):
        return None
    # prefer final.pt if present
    final = os.path.join(dir_path, "final.pt")
    if os.path.exists(final):
        return final
    cands = []
    for fn in os.listdir(dir_path):
        if fn.endswith(".pt") and (fn.startswith("step") or fn.startswith("interrupted_step")):
            m = re.findall(r"\d+", fn)
            if m:
                cands.append((int(m[0]), fn))
    if not cands:
        autosave = os.path.join(dir_path, "autosave.pt")
        return autosave if os.path.exists(autosave) else None
    cands.sort()
    return os.path.join(dir_path, cands[-1][1])

BASE_CKPT      = latest_ckpt("checkpoints_chat_v1") or r"checkpoints_chat_v1\final.pt"
OUT_DIR        = "finetuning"
LORA_ADAPTER   = os.path.join(OUT_DIR, "lora_adapter.pt")
LORA_FULLSTATE = os.path.join(OUT_DIR, "lora_full_state.pt")

os.makedirs(OUT_DIR, exist_ok=True)
FEEDBACK_DIR = "feedback"
FEEDBACK_SFT_JSONL = os.path.join(FEEDBACK_DIR, "feedback_sft_clean.jsonl") #"feedback_sft.jsonl")
os.makedirs(FEEDBACK_DIR, exist_ok=True)
SYNTH_SFT_JSONL = "data/synth_chat_sft.jsonl"
FEEDBACK_CKPT = os.path.join(FEEDBACK_DIR, "feedback_lora_ckpt.pt")
RAG_CACHE_DIR = "rag_cache"
os.makedirs(RAG_CACHE_DIR, exist_ok=True)

# ──────────────────────────────
# TRAINING — LoRA
# ──────────────────────────────
EPOCHS        = 3
BATCH_SIZE    = 32
GRAD_ACCUM    = 8
LEARNING_RATE = 1.5e-4
WARMUP_STEPS  = 300
GRAD_CLIP     = 1.0

PRINT_EVERY   = 20
SAMPLE_EVERY  = 200

MAX_NEW_TOKENS = 160
TEMPERATURE    = 0.7
TOP_P          = 0.9

# ──────────────────────────────
# LoRA CONFIG
# ──────────────────────────────
LORA_R      = 16
LORA_ALPHA  = 32
LORA_SCALE  = LORA_ALPHA / LORA_R

# target module name substrings (match your model)
LORA_TARGET = ["attn.qkv", "attn.proj"]

# ──────────────────────────────
# Helpers
# ──────────────────────────────
_space = re.compile(r"[ \t]+")
_many_blank_lines = re.compile(r"\n{3,}")

def normalize_preserve_newlines(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [_space.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _many_blank_lines.sub("\n\n", text)
    return text.strip()

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def now_ts() -> float:
    return time.time()

def build_prompt(instruction: str, style: str = "") -> str:
    # Consistent chat/instruction format (works for both chatty and tasky items)
    if style:
        return f"{style}\nUser: {instruction}\nAssistant:"
    return f"User: {instruction}\nAssistant:"

def build_prompt_with_context(instruction: str, context: str) -> str:
    if not context:
        return build_prompt(instruction)
    return (
        "Use the Context to answer. If the answer is not in Context, say you are unsure."
        "\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"User: {instruction}\nAssistant:"
    )

def split_prefix_response(tokenizer: spm.SentencePieceProcessor, instruction: str, output: str) -> Tuple[List[int], int]:
    """
    Returns full_ids padded later, plus prefix_len (tokens to mask).
    We mask loss on the prefix ("User: ... Assistant:") so only assistant content trains.
    """
    eos = tokenizer.eos_id()
    prompt = build_prompt(instruction)
    prefix_ids = tokenizer.encode(normalize_preserve_newlines(prompt), out_type=int)

    full_text = normalize_preserve_newlines(prompt + " " + output)
    full_ids = tokenizer.encode(full_text, out_type=int)
    if eos is not None:
        full_ids.append(eos)
    return full_ids, len(prefix_ids)

# ──────────────────────────────
# RAG (web search + embeddings)
#
# Design goals:
# - Keep the tiny model for smalltalk/tone.
# - For factual/unknown questions, use web retrieval + an extractive summarizer.
# - Never feed raw webpage menus into the model.

def _rag_slug(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _fetch_url_text(url: str, timeout: int = 10) -> str:
    """Fetch page text and aggressively strip navigation/boilerplate."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        return ""

    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "tiny-llm-rag"})
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "form", "button", "input"]):
            tag.decompose()

        # Drop common boilerplate containers.
        boiler_re = re.compile(r"(nav|menu|footer|header|cookie|consent|subscribe|login|signup|sidebar|toolbar|breadcrumb|promo|banner|ads)", re.I)
        for el in soup.find_all(attrs={"class": boiler_re}):
            el.decompose()
        for el in soup.find_all(attrs={"id": boiler_re}):
            el.decompose()

        # Prefer main/article bodies when available.
        root = soup.select_one("#mw-content-text") or soup.select_one("article") or soup.select_one("main") or soup.body or soup
        text = root.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)
    except Exception:
        return ""


def _chunk_text(text: str, max_chars: int = 600) -> List[str]:
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for para in text.split("\n"):
        if not para:
            continue
        if cur_len + len(para) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
        cur.append(para)
        cur_len += len(para) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _strip_menu_noise(text: str) -> str:
    drop_phrases = [
        "privacy", "cookies", "terms", "subscribe", "sign up", "sign in", "login",
        "all rights reserved", "contact", "about us", "newsletter",
        "skip to content", "skip to main content", "shopping cart",
        "click to expand", "click to open", "close icon", "caret", "accordion",
        "external links", "further reading", "retrieved from", "archived",
        "short description", "authority control", "wikiproject", "template:", "categories",
        "cookie policy", "privacy policy", "terms of service",
    ]

    out: List[str] = []
    for ln in (text or "").split("\n"):
        ln = ln.strip()
        if len(ln) < 4:
            continue
        low = ln.lower()

        if any(p in low for p in drop_phrases) and len(ln) < 140:
            continue
        if "http://" in low or "https://" in low:
            continue
        if "|" in ln:
            continue
        if low.count(">") >= 3:
            continue
        # Drop obvious nav/category lists.
        if len(re.findall(r"\b[A-Z][a-z]+\b", ln)) >= 10 and "." not in ln:
            continue

        out.append(ln)

    return "\n".join(out)


def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    # Normalize curly quotes to ASCII when they appear.
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", "\"").replace("\u201d", "\"")
    s = s.replace("\"", "")
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\(book\)", "", s, flags=re.I)
    s = re.sub(r"^\|\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _postprocess_smalltalk(ans: str) -> str:
    if not ans:
        return ans
    parts = re.split(r"(?<=[.!?])\s+", ans.strip())
    uniq: List[str] = []
    seen = set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
        if len(uniq) >= 2:
            break
    return " ".join(uniq).strip()


def _summarize_context(query: str, context: str, max_sentences: int = 4) -> str:
    """Generic extractive summary: pick highest-overlap sentences and strip boilerplate."""
    ctx = _strip_menu_noise(context)
    ctx = re.sub(r"\[[^\]]{1,40}\]", "", ctx)

    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if s.strip()]
    sentences: List[str] = []
    for s in raw:
        s = _clean_sentence(s)
        low = s.lower()
        if len(s) < 40 or len(s) > 280:
            continue
        if "http://" in low or "https://" in low:
            continue
        if "|" in s:
            continue
        if re.search(r"\b(britannica|bbc|wikipedia|magazine|newsletter|sign up|subscribe)\b", low) and len(s) < 160:
            continue
        if re.search(r"\b(press release|forward-looking|sec|shareholder|filings)\b", low):
            continue
        # Drop title-like lines (lots of Titlecase words, no verb).
        if len(re.findall(r"\b[A-Z][a-z]+\b", s)) >= 10 and not re.search(r"\b(is|are|was|were|causes|cause|means|because|delivers|provides|uses|scatters|scatter)\b", low):
            continue
        sentences.append(s)

    qlow = (query or "").lower()
    keywords = [w for w in re.findall(r"[a-z]{4,}", qlow) if w not in {"what", "which", "latest", "about", "why", "from", "that", "this"}]
    if not keywords:
        keywords = re.findall(r"[a-z]{4,}", qlow)

    scored: List[Tuple[int, str]] = []
    for s in sentences:
        score = sum(1 for k in keywords if k in s.lower())
        if score > 0:
            scored.append((score, s))

    if not scored:
        # Fallback: first decent sentence.
        for s in sentences:
            if len(s) >= 60:
                return s
        return ""

    scored.sort(reverse=True)
    selected: List[str] = []
    seen = set()
    for _, s in scored:
        key = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(s)
        if len(selected) >= max(2, min(max_sentences, 6)):
            break
    return " ".join(selected).strip()


class RagEngine:
    def __init__(self, cache_dir: str, model_name: str = "intfloat/e5-small-v2", timeout: int = 10):
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.timeout = timeout
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            # CPU is fine; keep GPU for the LM.
            self._embedder = SentenceTransformer(self.model_name, device="cpu")
        return self._embedder

    def retrieve(self, query: str, top_k: int = 6, web_k: int = 8, max_chunk_chars: int = 600, site: str = "") -> str:
        cache_key = _rag_slug((site + "|" + query).strip())
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_path):
            try:
                return Path(cache_path).read_text(encoding="utf-8")
            except Exception:
                pass

        try:
            from ddgs import DDGS
            import numpy as np
        except Exception:
            print("?? RAG deps missing. Install: pip install -U ddgs sentence-transformers requests beautifulsoup4")
            return ""

        q_in = (query or "").strip()
        q_low = q_in.lower()
        recency_intent = any(k in q_low for k in ["latest", "current", "today", "price", "release", "released", "news", "newest"])

        # Generic query rewrite for "latest/current" requests: bias toward recent docs.
        # (Not hardcoded to any single topic; only activates on recency intent.)
        if recency_intent:
            y = time.localtime().tm_year
            q_search = f"{q_in} {y} {y-1}"
            # If the user mentions NVIDIA/GPU families, add a generic "series" hint.
            if any(t in q_low for t in ["nvidia", "geforce", "gpu", "graphics"]):
                q_search += " series"
        else:
            q_search = q_in

        results = []
        try:
            with DDGS() as ddgs:
                q = f"site:{site} {q_search}" if site else q_search
                results = list(ddgs.text(q, max_results=web_k))
        except Exception:
            return ""

        docs: List[str] = []
        for r in results:
            url = r.get("href") or r.get("url")
            if not url:
                continue
            if site and site not in url:
                continue
            text = _fetch_url_text(url, timeout=self.timeout)
            if text:
                docs.append(_strip_menu_noise(text))

        if not docs:
            return ""

        chunks: List[str] = []
        for doc in docs:
            chunks.extend(_chunk_text(doc, max_chars=max_chunk_chars))
        if not chunks:
            return ""

        # If the user asked for "latest/current", downweight obviously old content.
        if recency_intent:
            y = time.localtime().tm_year
            filtered = []
            for ch in chunks:
                years = [int(v) for v in re.findall(r"\b(19\d{2}|20\d{2})\b", ch)]
                if years and max(years) <= (y - 5):
                    continue
                filtered.append(ch)
            if filtered:
                chunks = filtered

        embedder = self._get_embedder()
        q_emb = embedder.encode([query], normalize_embeddings=True)
        c_emb = embedder.encode(chunks, normalize_embeddings=True)
        scores = (q_emb @ c_emb.T)[0]

        if recency_intent:
            y = time.localtime().tm_year
            # Small generic boost for chunks mentioning recent years.
            for i, ch in enumerate(chunks):
                if str(y) in ch or str(y - 1) in ch:
                    scores[i] += 0.05

        top_idx = scores.argsort()[-top_k:][::-1]
        top_chunks = [chunks[int(i)] for i in top_idx]
        context = "\n\n".join(top_chunks)

        try:
            Path(cache_path).write_text(context, encoding="utf-8")
        except Exception:
            pass

        return context


def rag_extract_answer(query: str, context: str, max_sentences: int = 5) -> str:
    summary = _summarize_context(query, context, max_sentences=max_sentences)
    if summary:
        summary = re.sub(r"\[[^\]]*\]", "", summary)
        summary = re.sub(r"\s+", " ", summary).strip()
        return summary
    return "I'm not sure based on the retrieved sources."


def _normalize_for_overlap(text: str) -> List[str]:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.findall(r"[a-z]{3,}", t)


def is_uncertain_answer(user: str, draft: str, stats: dict) -> bool:
    u = set(_normalize_for_overlap(user))
    a = set(_normalize_for_overlap(draft))

    # Always RAG for "current" queries.
    u_low = (user or "").lower()
    if any(k in u_low for k in ["latest", "current", "today", "price", "release", "released", "202", "news"]):
        return True

    d = (draft or "").strip()
    if len(d) < 8 or len(d.split()) < 6:
        return True

    d_low = d.lower()
    if any(p in d_low for p in ["not sure", "i don't know", "unsure", "can't tell", "the best way to answer"]):
        return True

    # Low lexical overlap (ignore very short questions).
    if len(u) >= 4:
        overlap = len(u & a)
        if overlap < max(1, len(u) // 8):
            return True

    # Low model confidence (if stats provided).
    if stats and stats.get("avg_max_prob", 1.0) < 0.28:
        return True

    # Obvious nonsense: huge numbers or unrelated named entities without overlap.
    if re.search(r"\b\d{4,}\b", d_low) and len(u) >= 4 and len(u & a) == 0:
        return True

    return False


def is_smalltalk(text: str) -> bool:
    t = (text or "").lower().strip()
    t = re.sub(r"[^a-z0-9'\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True
    small = [
        "hi", "hello", "hey", "how are you", "good morning", "good evening",
        "good night", "thanks", "thank you", "bye", "what's up", "whats up",
        "how is your day", "how are you doing",
    ]
    return any(t == s or t.startswith(s + " ") for s in small)

def smalltalk_reply(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[^a-z0-9'\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return "Hi! How can I help?"
    if t.startswith(("hi", "hello", "hey")):
        return "Hi! How can I help?"
    if t.startswith("how are you"):
        return "I'm doing well—thanks for asking! How are you?"
    if t.startswith(("thanks", "thank you")):
        return "You're welcome!"
    if t.startswith(("bye", "goodbye")):
        return "Bye! Talk soon."
    if t.startswith(("what's up", "whats up")):
        return "Not much—how can I help?"
    if t.startswith("good morning"):
        return "Good morning! How can I help?"
    if t.startswith("good evening"):
        return "Good evening! How can I help?"
    if t.startswith("good night"):
        return "Good night! Sleep well."
    return ""


SMALLTALK_STYLE = (
    "You are a friendly chat assistant. Keep responses short, natural, and conversational. "
    "Avoid technical or random words."
)


def extract_summarize_text(user: str) -> str:
    u = (user or "").strip()
    lower = u.lower()
    for prefix in ("summarize this text:", "summarize:", "please summarize:"):
        if lower.startswith(prefix):
            return u[len(prefix):].strip()
    return ""


def try_simple_math(text: str) -> str:
    t = (text or "").strip().lower()
    m = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", t)
    if not m:
        return ""
    a = int(m.group(1))
    op = m.group(2)
    b = int(m.group(3))
    try:
        if op == "+":
            return str(a + b)
        if op == "-":
            return str(a - b)
        if op == "*":
            return str(a * b)
        if op == "/":
            if b == 0:
                return ""
            if a % b == 0:
                return str(a // b)
            return str(a / b)
    except Exception:
        return ""
    return ""

# Dataset
# Dataset
# ──────────────────────────────
class InstructionDataset(Dataset):
    def __init__(self, path: str, tokenizer: spm.SentencePieceProcessor, block_size: int):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.block_size = block_size
        self.samples = []
        pad_id = 0

        kept = 0
        for item in raw:
            instr = normalize_preserve_newlines(item.get("instruction", ""))
            out = normalize_preserve_newlines(item.get("output", ""))
            if not instr or not out:
                continue

            ids, prefix_len = split_prefix_response(tokenizer, instr, out)

            # drop too long examples (simple, stable)
            if len(ids) > block_size:
                continue

            # pad to block_size
            ids = ids + [pad_id] * (block_size - len(ids))

            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)

            # mask: only train assistant output
            # y index aligns with x positions; prefix_len includes "Assistant:" token(s)
            mask_upto = max(0, prefix_len - 1)
            y[:mask_upto] = -100
            y[y == pad_id] = -100

            self.samples.append((x, y))
            kept += 1

        print(f"📚 Loaded instruction samples: {kept:,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class FeedbackSFTDataset(Dataset):
    """
    Reads feedback JSONL where each row has:
      - instruction (str)
      - chosen (str)  # user-corrected assistant answer
    Trains only on the assistant part.
    """
    def __init__(self, jsonl_path: str, tokenizer: spm.SentencePieceProcessor, block_size: int):
        self.samples = []
        self.block_size = block_size
        pad_id = 0

        if not os.path.exists(jsonl_path):
            print(f"⚠️ No feedback file found at {jsonl_path}")
            return

        kept = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                instr = normalize_preserve_newlines(row.get("instruction", ""))
                chosen = normalize_preserve_newlines(row.get("chosen", ""))
                if not instr or not chosen:
                    continue

                ids, prefix_len = split_prefix_response(tokenizer, instr, chosen)

                if len(ids) > block_size:
                    continue

                ids = ids + [pad_id] * (block_size - len(ids))

                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(ids[1:], dtype=torch.long)

                mask_upto = max(0, prefix_len - 1)
                y[:mask_upto] = -100
                y[y == pad_id] = -100

                self.samples.append((x, y))
                kept += 1

        print(f"📝 Loaded feedback SFT examples: {kept:,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class SynthSFTDataset(Dataset):
    """
    Reads synthetic JSONL where each row has:
      - instruction (str)
      - output (str)
    Trains only on the assistant part.
    """
    def __init__(self, jsonl_path: str, tokenizer: spm.SentencePieceProcessor, block_size: int):
        self.samples = []
        self.block_size = block_size
        pad_id = 0

        if not os.path.exists(jsonl_path):
            print(f"⚠️ No synthetic file found at {jsonl_path}")
            return

        kept = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                instr = normalize_preserve_newlines(row.get("instruction", ""))
                out = normalize_preserve_newlines(row.get("output", ""))
                if not instr or not out:
                    continue

                ids, prefix_len = split_prefix_response(tokenizer, instr, out)

                if len(ids) > block_size:
                    continue

                ids = ids + [pad_id] * (block_size - len(ids))

                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(ids[1:], dtype=torch.long)

                mask_upto = max(0, prefix_len - 1)
                y[:mask_upto] = -100
                y[y == pad_id] = -100

                self.samples.append((x, y))
                kept += 1

        print(f"🧪 Loaded synthetic SFT examples: {kept:,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class FeedbackSFTIterableDataset(IterableDataset):
    """
    Streams feedback JSONL to avoid loading huge files into RAM.
    Each row has: instruction, chosen
    """
    def __init__(self, jsonl_path: str, tokenizer: spm.SentencePieceProcessor, block_size: int):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        pad_id = 0
        if not os.path.exists(self.jsonl_path):
            print(f"⚠️ No feedback file found at {self.jsonl_path}")
            return
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                instr = normalize_preserve_newlines(row.get("instruction", ""))
                chosen = normalize_preserve_newlines(row.get("chosen", ""))
                if not instr or not chosen:
                    continue

                ids, prefix_len = split_prefix_response(self.tokenizer, instr, chosen)
                if len(ids) > self.block_size:
                    continue

                ids = ids + [pad_id] * (self.block_size - len(ids))
                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(ids[1:], dtype=torch.long)

                mask_upto = max(0, prefix_len - 1)
                y[:mask_upto] = -100
                y[y == pad_id] = -100

                yield (x, y)

class SynthSFTIterableDataset(IterableDataset):
    """
    Streams synthetic JSONL to avoid loading huge files into RAM.
    Each row has: instruction, output
    """
    def __init__(self, jsonl_path: str, tokenizer: spm.SentencePieceProcessor, block_size: int):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        pad_id = 0
        if not os.path.exists(self.jsonl_path):
            print(f"⚠️ No synthetic file found at {self.jsonl_path}")
            return
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instr = normalize_preserve_newlines(row.get("instruction", ""))
                out = normalize_preserve_newlines(row.get("output", ""))
                if not instr or not out:
                    continue

                ids, prefix_len = split_prefix_response(self.tokenizer, instr, out)
                if len(ids) > self.block_size:
                    continue

                ids = ids + [pad_id] * (self.block_size - len(ids))
                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(ids[1:], dtype=torch.long)

                mask_upto = max(0, prefix_len - 1)
                y[:mask_upto] = -100
                y[y == pad_id] = -100

                yield (x, y)

# ──────────────────────────────
# Model (same as your base training)
# ──────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.proj_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class FastGPT(nn.Module):
    def __init__(self, vocab, dim, heads, layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(block_size, dim)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

        print(f"🔧 Model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.shape[1] > self.block_size:
            x = x[:, -self.block_size:]
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok(x) + self.pos(pos))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))

# ──────────────────────────────
# LoRA injection
# ──────────────────────────────
def inject_lora(model: nn.Module, device: str):
    injected = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in LORA_TARGET):
            in_f, out_f = module.in_features, module.out_features

            module.lora_A = nn.Parameter(torch.zeros(LORA_R, in_f, device=device))
            module.lora_B = nn.Parameter(torch.zeros(out_f, LORA_R, device=device))
            module.lora_scale = LORA_SCALE

            nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B)

            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False

            orig_forward = module.forward

            def forward(x, orig_forward=orig_forward, m=module):
                # x: [*, in_f]
                return orig_forward(x) + ((x @ m.lora_A.T) @ m.lora_B.T) * m.lora_scale

            module.forward = forward
            injected += 1

    print(f"✓ LoRA injected into {injected} linear layers")

def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            sd[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            sd[f"{name}.lora_scale"] = torch.tensor(float(module.lora_scale))
    return sd

def load_lora_state_dict(model: nn.Module, lora_sd: Dict[str, torch.Tensor], device: str):
    for name, module in model.named_modules():
        keyA = f"{name}.lora_A"
        keyB = f"{name}.lora_B"
        keyS = f"{name}.lora_scale"
        if keyA in lora_sd and hasattr(module, "lora_A"):
            module.lora_A.data.copy_(lora_sd[keyA].to(device))
            module.lora_B.data.copy_(lora_sd[keyB].to(device))
            module.lora_scale = float(lora_sd.get(keyS, module.lora_scale))

# ──────────────────────────────
# Load base checkpoint
# ──────────────────────────────
def load_base_checkpoint(model: nn.Module, path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    print(f"✓ Loaded base checkpoint: {path}")

# ──────────────────────────────
# Sampling (top-p)
# ──────────────────────────────
@torch.no_grad()
def sample_top_p(probs: torch.Tensor, top_p: float) -> int:
    # probs: [V]
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    cutoff = torch.searchsorted(cumsum, torch.tensor(top_p, device=probs.device))
    cutoff = int(cutoff.item())
    cutoff = max(1, min(cutoff + 1, sorted_probs.numel()))
    filtered_probs = sorted_probs[:cutoff]
    filtered_idx = sorted_idx[:cutoff]
    filtered_probs = filtered_probs / filtered_probs.sum()
    next_i = torch.multinomial(filtered_probs, 1).item()
    return int(filtered_idx[next_i].item())

@torch.no_grad()
def generate(model, sp, device, user_text: str, history: List[Tuple[str, str]] = None, rag_context: str = "", return_stats: bool = False, style: str = ""):
    model.eval()
    history = history or []

    # Simple rolling context
    ctx = ""
    for u, a in history[-6:]:
        ctx += f"User: {u}\nAssistant: {a}\n"
    if rag_context:
        ctx += (
            "Use the Context to answer. If the answer is not in Context, say you are unsure."
            "\n\n"
            f"Context:\n{rag_context}\n\n"
        )
    if style:
        ctx += f"{style}\n"
    ctx += f"User: {user_text}\nAssistant:"

    ids = sp.encode(normalize_preserve_newlines(ctx), out_type=int)
    x = torch.tensor([ids], device=device)

    eos = sp.eos_id()
    max_probs = []
    for _ in range(MAX_NEW_TOKENS):
        temp = TEMPERATURE if not rag_context else 0.2
        logits = model(x[:, -BLOCK_SIZE:])[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1).squeeze(0)
        top_p = TOP_P if not rag_context else 0.8
        nxt = sample_top_p(probs, top_p)
        max_probs.append(float(probs.max().item()))
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if eos is not None and nxt == eos:
            break

    text = sp.decode(x[0].tolist())
    # Return only the last assistant segment
    # (best-effort split)
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1]
    out = text.strip()
    if return_stats:
        avg_max_prob = sum(max_probs) / max(1, len(max_probs))
        return out, {"avg_max_prob": avg_max_prob, "tokens": len(max_probs)}
    return out

# ──────────────────────────────
# LoRA train
# ──────────────────────────────
def run_lora():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    vocab = sp.get_piece_size()
    print(f"📚 Vocab size: {vocab:,} | eos_id={sp.eos_id()}")

    ds = InstructionDataset(INSTRUCT_JSON, sp, BLOCK_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = FastGPT(vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(device)
    load_base_checkpoint(model, BASE_CKPT, device)

    # freeze base
    for p in model.parameters():
        p.requires_grad = False

    inject_lora(model, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"🧠 Trainable params: {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.0)
    sched = LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / WARMUP_STEPS))

    scaler = torch.amp.GradScaler("cuda") if (device == "cuda") else None

    step = 0
    model.train()

    test_prompts = [
        "How are you?",
        "I feel stressed. Any quick tips?",
        "What is a transformer model?",
        "Explain gradient descent simply.",
    ]

    try:
        for epoch in range(EPOCHS):
            pbar = tqdm(dl, desc=f"LoRA Epoch {epoch+1}/{EPOCHS}")
            loss_acc = 0.0

            for i, (x, y) in enumerate(pbar):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if (i % GRAD_ACCUM) == 0:
                    opt.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=-100) / GRAD_ACCUM
                    scaler.scale(loss).backward()
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=-100) / GRAD_ACCUM
                    loss.backward()

                if ((i + 1) % GRAD_ACCUM) == 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                        opt.step()

                    sched.step()
                    step += 1
                    loss_acc += loss.item() * GRAD_ACCUM

                    if step % PRINT_EVERY == 0:
                        avg = loss_acc / PRINT_EVERY
                        ppl = math.exp(min(avg, 20))
                        pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.2f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                        loss_acc = 0.0

                    if step % SAMPLE_EVERY == 0:
                        q = test_prompts[(step // SAMPLE_EVERY) % len(test_prompts)]
                        out = generate(model, sp, device, q, history=[])
                        print("\n💬 SAMPLE")
                        print(f"User: {q}\nAssistant: {out}\n")

        # Save LoRA only
        torch.save(lora_state_dict(model), LORA_ADAPTER)
        print(f"💾 Saved LoRA adapter: {LORA_ADAPTER}")

        # Save full state_dict (base + LoRA) for convenience
        torch.save(model.state_dict(), LORA_FULLSTATE)
        print(f"💾 Saved full (base+LoRA) state_dict: {LORA_FULLSTATE}")

        print("✅ LoRA fine-tuning complete")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted LoRA. Saving adapter so far...")
        torch.save(lora_state_dict(model), LORA_ADAPTER)
        print(f"💾 Saved LoRA adapter: {LORA_ADAPTER}")

def run_feedback_lora():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    vocab = sp.get_piece_size()
    print(f"📚 Vocab size: {vocab:,} | eos_id={sp.eos_id()}")

    ds = FeedbackSFTIterableDataset(FEEDBACK_SFT_JSONL, sp, BLOCK_SIZE)
    # IterableDataset can't be shuffled by DataLoader; keep deterministic order
    dl = DataLoader(ds, batch_size=min(BATCH_SIZE, 8), shuffle=False, num_workers=0, pin_memory=True)

    model = FastGPT(vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(device)
    load_base_checkpoint(model, BASE_CKPT, device)

    # Freeze base
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA and (optionally) load existing adapter to continue improving
    inject_lora(model, device)
    if os.path.exists(LORA_ADAPTER):
        lora_sd = torch.load(LORA_ADAPTER, map_location="cpu")
        load_lora_state_dict(model, lora_sd, device)
        print(f"✓ Loaded existing LoRA adapter: {LORA_ADAPTER}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.0)
    sched = LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / max(50, WARMUP_STEPS // 3)))

    scaler = torch.amp.GradScaler("cuda") if (device == "cuda") else None

    # Keep this short to avoid overfitting feedback
    FEEDBACK_EPOCHS = 2
    SAVE_EVERY_STEPS = 100

    step = 0
    start_epoch = 0
    steps_to_skip = 0
    model.train()

    try:
        if os.path.exists(FEEDBACK_CKPT):
            ckpt = torch.load(FEEDBACK_CKPT, map_location="cpu")
            if "lora" in ckpt:
                load_lora_state_dict(model, ckpt["lora"], device)
            if "opt" in ckpt:
                opt.load_state_dict(ckpt["opt"])
            if "sched" in ckpt:
                sched.load_state_dict(ckpt["sched"])
            if "scaler" in ckpt and scaler is not None:
                scaler.load_state_dict(ckpt["scaler"])
            step = int(ckpt.get("step", 0))
            start_epoch = int(ckpt.get("epoch", 0))
            steps_to_skip = int(ckpt.get("steps_in_epoch", 0))
            print(f"↩️ Resuming feedback LoRA at epoch {start_epoch+1}, step {step}")

        for epoch in range(start_epoch, FEEDBACK_EPOCHS):
            pbar = tqdm(dl, desc=f"Feedback LoRA Epoch {epoch+1}/{FEEDBACK_EPOCHS}")
            loss_acc = 0.0

            for i, (x, y) in enumerate(pbar):
                if steps_to_skip > 0:
                    steps_to_skip -= 1
                    continue
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if (i % GRAD_ACCUM) == 0:
                    opt.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=-100) / GRAD_ACCUM
                    scaler.scale(loss).backward()
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=-100) / GRAD_ACCUM
                    loss.backward()

                if ((i + 1) % GRAD_ACCUM) == 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                        opt.step()
                    sched.step()

                    step += 1
                    loss_acc += loss.item() * GRAD_ACCUM

                    if step % PRINT_EVERY == 0:
                        avg = loss_acc / PRINT_EVERY
                        ppl = math.exp(min(avg, 20))
                        pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.2f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                        loss_acc = 0.0

                    if step % SAVE_EVERY_STEPS == 0:
                        torch.save({
                            "lora": lora_state_dict(model),
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict(),
                            "scaler": scaler.state_dict() if scaler is not None else None,
                            "epoch": epoch,
                            "step": step,
                            "steps_in_epoch": i + 1,
                        }, FEEDBACK_CKPT)
                        print(f"💾 Saved feedback LoRA checkpoint: {FEEDBACK_CKPT}")

        torch.save(lora_state_dict(model), LORA_ADAPTER)
        print(f"💾 Updated LoRA adapter saved: {LORA_ADAPTER}")
        torch.save(model.state_dict(), LORA_FULLSTATE)
        print(f"💾 Updated full (base+LoRA) state saved: {LORA_FULLSTATE}")
        if os.path.exists(FEEDBACK_CKPT):
            os.remove(FEEDBACK_CKPT)

        print("✅ Feedback LoRA complete")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted feedback LoRA. Saving adapter so far...")
        torch.save({
            "lora": lora_state_dict(model),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "steps_in_epoch": i + 1 if "i" in locals() else 0,
        }, FEEDBACK_CKPT)
        print(f"💾 Saved feedback LoRA checkpoint: {FEEDBACK_CKPT}")
        torch.save(lora_state_dict(model), LORA_ADAPTER)
        print(f"💾 Saved LoRA adapter: {LORA_ADAPTER}")

def run_synth_lora():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    vocab = sp.get_piece_size()
    print(f"📚 Vocab size: {vocab:,} | eos_id={sp.eos_id()}")

    ds = SynthSFTIterableDataset(SYNTH_SFT_JSONL, sp, BLOCK_SIZE)
    # IterableDataset can't be shuffled by DataLoader; keep deterministic order
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = FastGPT(vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(device)
    load_base_checkpoint(model, BASE_CKPT, device)

    # Freeze base
    for p in model.parameters():
        p.requires_grad = False

    inject_lora(model, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"🧠 Trainable params: {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.0)
    sched = LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / WARMUP_STEPS))

    scaler = torch.amp.GradScaler("cuda") if (device == "cuda") else None

    step = 0
    model.train()

    try:
        for epoch in range(EPOCHS):
            pbar = tqdm(dl, desc=f"Synth LoRA Epoch {epoch+1}/{EPOCHS}")
            loss_acc = 0.0

            for i, (x, y) in enumerate(pbar):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if (i % GRAD_ACCUM) == 0:
                    opt.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=-100) / GRAD_ACCUM
                    scaler.scale(loss).backward()
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=-100) / GRAD_ACCUM
                    loss.backward()

                if ((i + 1) % GRAD_ACCUM) == 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                        opt.step()

                    sched.step()
                    step += 1
                    loss_acc += loss.item() * GRAD_ACCUM

                    if step % PRINT_EVERY == 0:
                        avg = loss_acc / PRINT_EVERY
                        ppl = math.exp(min(avg, 20))
                        pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.2f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                        loss_acc = 0.0

        torch.save(lora_state_dict(model), LORA_ADAPTER)
        print(f"💾 Saved LoRA adapter: {LORA_ADAPTER}")
        torch.save(model.state_dict(), LORA_FULLSTATE)
        print(f"💾 Saved full (base+LoRA) state: {LORA_FULLSTATE}")

        print("✅ Synth LoRA complete")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted synth LoRA. Saving adapter so far...")
        torch.save(lora_state_dict(model), LORA_ADAPTER)
        print(f"💾 Saved LoRA adapter: {LORA_ADAPTER}")

# ──────────────────────────────
# Chat CLI
# ──────────────────────────────
def run_chat(use_lora: bool, rag_mode: str, rag_top_k: int, rag_web_k: int, rag_chunk_chars: int, rag_site: str, rag_debug: bool, rag_extract: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    vocab = sp.get_piece_size()

    model = FastGPT(vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(device)
    load_base_checkpoint(model, BASE_CKPT, device)

    # Freeze base (not required for chat, but fine)
    for p in model.parameters():
        p.requires_grad = False

    inject_lora(model, device)  # inject structure so we can load adapter

    if use_lora:
        if not os.path.exists(LORA_ADAPTER):
            raise FileNotFoundError(f"LoRA adapter not found: {LORA_ADAPTER}")
        lora_sd = torch.load(LORA_ADAPTER, map_location="cpu")
        load_lora_state_dict(model, lora_sd, device)
        print(f"✓ Loaded LoRA adapter: {LORA_ADAPTER}")
    else:
        print("ℹ️ Running WITHOUT LoRA (base only)")

    # Lazy-load embedder only if we actually enter the RAG path.
    rag = RagEngine(RAG_CACHE_DIR) if rag_mode != "off" else None

    print("\nType messages. Commands: /reset, /exit\n")
    history: List[Tuple[str, str]] = []

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user:
            continue
        if user.lower() in ("/exit", "/quit"):
            break
        if user.lower() == "/reset":
            history = []
            print("✓ Reset history.\n")
            continue

        # Summarize mode: extractive summary from user-provided text
        summarize_text = extract_summarize_text(user)
        if summarize_text:
            summary = _summarize_context("summarize", summarize_text, max_sentences=2)
            if not summary:
                summary = "I'm not sure how to summarize that."
            ans = summary
            print(f"Bot: {ans}\n")
            history.append((user, ans))
            cmd = input("Rate: [enter=skip, u=up, e=edit, q=quit] ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "u":
                append_jsonl(FEEDBACK_SFT_JSONL, {
                    "ts": now_ts(),
                    "instruction": user,
                    "model": ans,
                    "chosen": ans,
                    "tag": "upvote",
                })
                print("✓ Logged upvote.\n")
            elif cmd == "e":
                print("Paste your preferred answer. End with an empty line:")
                lines = []
                while True:
                    ln = input()
                    if ln == "":
                        break
                    lines.append(ln)
                chosen = normalize_preserve_newlines("\n".join(lines))
                if chosen:
                    append_jsonl(FEEDBACK_SFT_JSONL, {
                        "ts": now_ts(),
                        "instruction": user,
                        "model": ans,
                        "chosen": chosen,
                        "tag": "edited",
                    })
                    print("✓ Logged corrected answer.\n")
                else:
                    print("⚠️ Empty correction, skipped.\n")
            continue

        math_ans = try_simple_math(user)
        if math_ans:
            ans = f"The answer is {math_ans}."
            print(f"Bot: {ans}\n")
            history.append((user, ans))
            # Feedback UI
            cmd = input("Rate: [enter=skip, u=up, e=edit, q=quit] ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "u":
                append_jsonl(FEEDBACK_SFT_JSONL, {
                    "ts": now_ts(),
                    "instruction": user,
                    "model": ans,
                    "chosen": ans,
                    "tag": "upvote",
                })
                print("✓ Logged upvote.\n")
            elif cmd == "e":
                print("Paste your preferred answer. End with an empty line:")
                lines = []
                while True:
                    ln = input()
                    if ln == "":
                        break
                    lines.append(ln)
                chosen = normalize_preserve_newlines("\n".join(lines))
                if chosen:
                    append_jsonl(FEEDBACK_SFT_JSONL, {
                        "ts": now_ts(),
                        "instruction": user,
                        "model": ans,
                        "chosen": chosen,
                        "tag": "edited",
                    })
                    print("✓ Logged corrected answer.\n")
                else:
                    print("⚠️ Empty correction, skipped.\n")
            continue

        if is_smalltalk(user):
            canned = smalltalk_reply(user)
            if canned:
                ans = canned
                print(f"Bot: {ans}\n")
                history.append((user, ans))
                cmd = input("Rate: [enter=skip, u=up, e=edit, q=quit] ").strip().lower()
                if cmd == "q":
                    break
                elif cmd == "u":
                    append_jsonl(FEEDBACK_SFT_JSONL, {
                        "ts": now_ts(),
                        "instruction": user,
                        "model": ans,
                        "chosen": ans,
                        "tag": "upvote",
                    })
                    print("✓ Logged upvote.\n")
                elif cmd == "e":
                    print("Paste your preferred answer. End with an empty line:")
                    lines = []
                    while True:
                        ln = input()
                        if ln == "":
                            break
                        lines.append(ln)
                    chosen = normalize_preserve_newlines("\n".join(lines))
                    if chosen:
                        append_jsonl(FEEDBACK_SFT_JSONL, {
                            "ts": now_ts(),
                            "instruction": user,
                            "model": ans,
                            "chosen": chosen,
                            "tag": "edited",
                        })
                        print("✓ Logged corrected answer.\n")
                    else:
                        print("⚠️ Empty correction, skipped.\n")
                continue

        # First try: model-only draft (used for non-smalltalk router)
        draft, stats = generate(model, sp, device, user, history=history, rag_context="", return_stats=True, style="")

        if is_smalltalk(user):
            ans = _postprocess_smalltalk(draft)
        else:
            want_rag = False
            if rag_mode == "always":
                want_rag = True
            elif rag_mode == "auto":
                want_rag = is_uncertain_answer(user, draft, stats)

            context = ""
            if want_rag and rag is not None:
                context = rag.retrieve(
                    user,
                    top_k=rag_top_k,
                    web_k=rag_web_k,
                    max_chunk_chars=rag_chunk_chars,
                    site=rag_site,
                )
                if rag_debug and context:
                    print("\n[rag] context snippet:")
                    print(context[:1200].strip() + "\n")

            if want_rag and rag_extract and context:
                ans = rag_extract_answer(user, context, max_sentences=6)
            elif want_rag and context:
                # If extract is disabled, we still keep the model on short, cleaned context.
                ans = generate(model, sp, device, user, history=history, rag_context=context)
            else:
                ans = draft
        print(f"Bot: {ans}\n")
        history.append((user, ans))

        # Feedback UI
        # Enter = skip, u = upvote, e = edit/correct, q = quit
        cmd = input("Rate: [enter=skip, u=up, e=edit, q=quit] ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "u":
            append_jsonl(FEEDBACK_SFT_JSONL, {
                "ts": now_ts(),
                "instruction": user,
                "model": ans,
                "chosen": ans,   # keep as-is
                "tag": "upvote",
            })
            print("✓ Logged upvote.\n")
        elif cmd == "e":
            print("Paste your preferred answer. End with an empty line:")
            lines = []
            while True:
                ln = input()
                if ln == "":
                    break
                lines.append(ln)
            chosen = normalize_preserve_newlines("\n".join(lines))
            if chosen:
                append_jsonl(FEEDBACK_SFT_JSONL, {
                    "ts": now_ts(),
                    "instruction": user,
                    "model": ans,
                    "chosen": chosen,
                    "tag": "edited",
                })
                print("✓ Logged corrected answer.\n")
            else:
                print("⚠️ Empty correction, skipped.\n")

# ──────────────────────────────
# Main
# ──────────────────────────────
def main():
    global BASE_CKPT, INSTRUCT_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lora", "feedback_lora", "synth_lora", "chat"], required=True)
    ap.add_argument("--use_lora", action="store_true", help="(chat mode) load LoRA adapter")
    ap.add_argument("--base_ckpt", default=BASE_CKPT, help="base checkpoint path")
    ap.add_argument("--instruct_json", default=INSTRUCT_JSON, help="instruction json path")
    ap.add_argument("--rag_mode", choices=["auto", "always", "off"], default="auto", help="(chat mode) when to use RAG")
    ap.add_argument("--rag_top_k", type=int, default=6, help="(chat mode) top chunks to use")
    ap.add_argument("--rag_web_k", type=int, default=8, help="(chat mode) web results to fetch")
    ap.add_argument("--rag_chunk_chars", type=int, default=600, help="(chat mode) max chars per chunk")
    ap.add_argument("--rag_site", default="", help="(chat mode) restrict search to a site, e.g. wikipedia.org")
    ap.add_argument("--rag_debug", action="store_true", help="(chat mode) print retrieved context")
    ap.add_argument("--rag_no_extract", action="store_true", help="(chat mode) use model instead of extractive answer")
    args = ap.parse_args()
    BASE_CKPT = args.base_ckpt
    INSTRUCT_JSON = args.instruct_json

    if args.mode == "lora":
        run_lora()
    elif args.mode == "feedback_lora":
        run_feedback_lora()
    elif args.mode == "synth_lora":
        run_synth_lora()
    else:
        run_chat(
            use_lora=args.use_lora,
            rag_mode=args.rag_mode,
            rag_top_k=args.rag_top_k,
            rag_web_k=args.rag_web_k,
            rag_chunk_chars=args.rag_chunk_chars,
            rag_site=args.rag_site,
            rag_debug=args.rag_debug,
            rag_extract=not args.rag_no_extract,
        )

if __name__ == "__main__":
    main()
