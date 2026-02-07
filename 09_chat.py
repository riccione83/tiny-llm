#!/usr/bin/env python3
"""
Chat CLI for the from-scratch summarizer model (base + optional LoRA).

Focus: user-pasted summarization:
  Summarize: <text>

This CLI:
- Loads a base checkpoint
- Optionally loads a LoRA adapter (only injects LoRA if adapter is provided)
- Normalizes "Summarize:" / "Summarize in 2 sentences:" into the exact SFT format:
    "Summarize in 2 sentences:\n<text>"
- Uses near-greedy decoding by default (helps evaluation clarity)
"""

import argparse
import email.utils
import html
import json
import math
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


BLOCK_SIZE = 768
EMBED_DIM = 896
NUM_HEADS = 14
NUM_LAYERS = 16
DROPOUT = 0.0

MAX_NEW_TOKENS = 180
DEFAULT_TEMPERATURE = 0.0   # deterministic by default
DEFAULT_TOP_P = 1.0         # deterministic with top-1 path below
WEB_FETCH_TIMEOUT = 10

URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
BAD_SENTENCE_HINTS = [
    "jump to content",
    "main menu",
    "personal tools",
    "donate",
    "create account",
    "log in",
    "search",
    "navigation",
    "privacy policy",
    "terms",
    "published time",
    "url source",
    "markdown content",
    "related articles",
    "references",
]

GENERIC_NONANSWER_HINTS = [
    "how can i help",
    "i'm doing well",
    "i am doing well",
    "happy to help",
    "no problem",
]

QUERY_STOPWORDS = {
    "what", "who", "when", "where", "which", "how", "are", "is", "the", "and", "or", "of", "in", "to",
    "for", "a", "an", "this", "that", "about", "latest", "recent", "new", "news", "current", "year",
    "questo", "questa", "sono", "nel", "nella", "della", "delle", "degli", "gli", "le", "il", "lo",
}

TRUSTED_DOMAIN_HINTS = [
    "wikipedia.org",
    ".gov",
    ".edu",
    "reuters.com",
    "bbc.com",
    "apnews.com",
    "bloomberg.com",
    "ft.com",
    "ec.europa.eu",
    "europa.eu",
    "nasa.gov",
    "nih.gov",
    "nature.com",
    "science.org",
    "who.int",
]

LOW_QUALITY_DOMAIN_HINTS = [
    "medium.com",
    "linkedin.com",
    "quora.com",
    "pinterest.",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
]



def normalize(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonicalize_user_message(user: str) -> str:
    """
    Convert common user formats into the exact training instruction:

      Training instruction:
        Summarize in 2 sentences:
        <text>

    Supported inputs:
      - "Summarize: <text>"
      - "Summarize in 2 sentences: <text>"
      - "summarize: <text>" (case-insensitive)
    """
    u = user.strip()
    low = u.lower()

    # Match "Summarize in 2 sentences:" prefix (including "exactly")
    if low.startswith("summarize in exactly 2 sentences") or low.startswith("summarize in exactly two sentences"):
        body = u.split(":", 1)[1].strip() if ":" in u else u.split("sentences", 1)[1].strip()
        return "Summarize in 2 sentences:\n" + body
    if low.startswith("summarize in 2 sentences:"):
        body = u.split(":", 1)[1].strip()
        return "Summarize in 2 sentences:\n" + body

    # Match "Summarize:" prefix
    if low.startswith("summarize:"):
        body = u.split(":", 1)[1].strip()
        return "Summarize in 2 sentences:\n" + body

    # Italian summarization variants
    if low.startswith("riassumi in 2 frasi:") or low.startswith("riassumi in due frasi:") or low.startswith("riassumi in esattamente 2 frasi:"):
        body = u.split(":", 1)[1].strip()
        return "Summarize in 2 sentences:\n" + body
    if low.startswith("riassumi:"):
        body = u.split(":", 1)[1].strip()
        return "Summarize in 2 sentences:\n" + body
    if low.startswith("cosa dice questo articolo:"):
        body = u.split(":", 1)[1].strip()
        return "Summarize in 2 sentences:\n" + body
    if low.startswith("what does this article say:"):
        body = u.split(":", 1)[1].strip()
        return "Summarize in 2 sentences:\n" + body

    # If user already pasted something like the exact format, keep it.
    if low.startswith("summarize in 2 sentences") and "\n" in u:
        return "Summarize in 2 sentences:\n" + u.split("\n", 1)[1].strip()

    return u


def _find_first_url(text: str) -> Optional[str]:
    m = URL_RE.search(text or "")
    return m.group(0).strip() if m else None


def _strip_html(raw: str) -> str:
    t = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
    t = re.sub(r"(?is)<[^>]+>", " ", t)
    t = html.unescape(t)
    return normalize(t)


def _fetch_text(url: str, timeout: int = 15) -> Optional[str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; tiny-llm-bot/1.0)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    if "<html" in raw.lower():
        return _strip_html(raw)
    return normalize(raw)


def _fetch_raw_html(url: str, timeout: int = 15) -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _jina_proxy_url(url: str) -> str:
    p = urllib.parse.urlparse(url)
    if not p.scheme:
        p = urllib.parse.urlparse("https://" + url)
    target = p.netloc + p.path
    if p.query:
        target += "?" + p.query
    return "https://r.jina.ai/http://" + target


def _fetch_article_text(url: str, timeout: int = 15) -> Optional[str]:
    # Try readability proxy first, then direct fetch fallback.
    proxy = _jina_proxy_url(url)
    txt = _fetch_text(proxy, timeout=timeout)
    if txt and len(txt) > 120:
        return txt
    return _fetch_text(url, timeout=timeout)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for p in parts:
        p = p.strip(" \t\n-•")
        if not p:
            continue
        if len(p.split()) < 3:
            continue
        out.append(p)
    return out


def _clean_web_noise(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if "Markdown Content:" in t:
        t = t.split("Markdown Content:", 1)[1]
    t = html.unescape(t)
    t = re.sub(r"\[\[[^\]]+\]\]\([^)]+\)", " ", t)           # drop wiki-style ref links
    t = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", t)              # drop markdown images
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)           # keep markdown link text
    t = re.sub(r"\[\d+\]", " ", t)                           # drop citation markers like [5]
    t = re.sub(r"\[\[[^\]]+\]\]", " ", t)                    # drop raw wiki markers [[a]]
    t = re.sub(r"https?://\S+", " ", t)                      # drop raw URLs
    t = re.sub(r"^\s*[-*]\s+", "", t, flags=re.MULTILINE)    # drop bullet markers
    t = re.sub(r"^\s*=+\s*$", " ", t, flags=re.MULTILINE)    # drop markdown heading bars

    noise_patterns = [
        r"title\s*:\s*[^\n]+",
        r"url source\s*:\s*[^\n]+",
        r"published time\s*:\s*[^\n]+",
        r"markdown content\s*:",
        r"home\s*\|\s*news\s*\|\s*contact",
        r"home\s*\|\s*about\s*\|\s*contact",
        r"related articles?",
        r"privacy policy",
        r"\bterms\b",
        r"source:\s*[^.!\n]+",
    ]
    for pat in noise_patterns:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    return normalize(t)


def _is_content_sentence(s: str) -> bool:
    low = s.lower().strip()
    if len(low.split()) < 5:
        return False
    if "http://" in low or "https://" in low or "www." in low:
        return False
    if any(hint in low for hint in BAD_SENTENCE_HINTS):
        return False
    alpha = sum(ch.isalpha() for ch in low)
    if alpha < max(15, int(len(low) * 0.4)):
        return False
    return True


def _best_sentence_pair(text: str) -> List[str]:
    sents = _split_sentences(text)
    if not sents:
        return []
    good = [s for s in sents if _is_content_sentence(s)]
    src = good if len(good) >= 2 else sents
    return _ensure_two_units(src)


def _ensure_two_units(items: List[str]) -> List[str]:
    if len(items) >= 2:
        return items[:2]
    if not items:
        return []
    one = items[0].strip()
    clauses = re.split(r"\s*(?:;|,| but | and )\s*", one, maxsplit=1, flags=re.IGNORECASE)
    if len(clauses) == 2 and len(clauses[0].split()) >= 3 and len(clauses[1].split()) >= 3:
        a = clauses[0].strip().rstrip(".") + "."
        b = clauses[1].strip().rstrip(".") + "."
        return [a, b]
    return [one, one]


def try_extractive_summary_reply(user: str) -> Optional[str]:
    u = (user or "").strip()
    low = u.lower()

    mode = None
    body = ""
    if low.startswith("summarize as exactly 2 bullet points:") or low.startswith("summarize as exactly two bullet points:"):
        mode = "bullet"
        body = u.split(":", 1)[1].strip()
    elif low.startswith("summarize in exactly 2 sentences") or low.startswith("summarize in exactly two sentences"):
        mode = "two"
        body = u.split(":", 1)[1].strip() if ":" in u else u.split("sentences", 1)[1].strip()
    elif low.startswith("summarize in 2 sentences:"):
        mode = "two"
        body = u.split(":", 1)[1].strip()
    elif low.startswith("summarize:"):
        mode = "two"
        body = u.split(":", 1)[1].strip()
    elif low.startswith("riassumi come esattamente 2 punti elenco:") or low.startswith("riassumi come 2 punti elenco:"):
        mode = "bullet"
        body = u.split(":", 1)[1].strip()
    elif low.startswith("riassumi in 2 frasi:") or low.startswith("riassumi in due frasi:") or low.startswith("riassumi in esattamente 2 frasi:"):
        mode = "two"
        body = u.split(":", 1)[1].strip()
    elif low.startswith("riassumi:"):
        mode = "two"
        body = u.split(":", 1)[1].strip()
    elif low.startswith("cosa dice questo articolo:"):
        mode = "two"
        body = u.split(":", 1)[1].strip()
    elif low.startswith("what does this article say:"):
        mode = "two"
        body = u.split(":", 1)[1].strip()

    if mode is None:
        return None

    body = _clean_web_noise(body)
    if not body:
        return None

    # Keep this fallback for short/medium pasted text to reduce hallucinations.
    if len(body) > 1400:
        return None

    pair = _best_sentence_pair(body)
    if not pair:
        return None

    if mode == "bullet":
        return f"- {pair[0]}\n- {pair[1]}"
    return f"{pair[0]} {pair[1]}".strip()


def _summarize_plain_text(text: str, bullets: bool = False) -> Optional[str]:
    clean = _clean_web_noise(text)
    if not clean:
        return None
    pair = _best_sentence_pair(clean)
    if not pair:
        return None
    if bullets:
        return f"- {pair[0]}\n- {pair[1]}"
    return f"{pair[0]} {pair[1]}".strip()


def _sentence_count(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    parts = [p for p in parts if p.strip()]
    return len(parts)


def _two_sentence_exact(answer: str) -> bool:
    return _sentence_count(answer) == 2


def _exactly_two_bullets(answer: str) -> bool:
    lines = [ln for ln in (answer or "").splitlines() if ln.strip()]
    if len(lines) != 2:
        return False
    return all(ln.lstrip().startswith("-") for ln in lines)


def _is_number_only(answer: str) -> bool:
    return re.fullmatch(r"-?\d+(?:\.\d+)?", (answer or "").strip()) is not None


def _is_yes_no_only(answer: str) -> bool:
    return (answer or "").strip().upper() in {"YES", "NO", "SI"}


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t.strip())
    return t.strip()


def _is_json_only(answer: str) -> bool:
    t = _strip_code_fence(answer)
    if not t:
        return False
    if not (t.startswith("{") or t.startswith("[")):
        return False
    try:
        obj = json.loads(t)
    except Exception:
        return False
    return isinstance(obj, (dict, list))


def _extract_first_number(text: str) -> Optional[str]:
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return m.group(0) if m else None


def _extract_yes_no(text: str) -> Optional[str]:
    low = (text or "").lower()
    if re.search(r"\byes\b", low):
        return "YES"
    if re.search(r"\bno\b", low):
        return "NO"
    if re.search(r"\btrue\b", low):
        return "YES"
    if re.search(r"\bfalse\b", low):
        return "NO"
    if re.search(r"\bsi\b", low):
        return "SI"
    return None


def _extract_json_blob(text: str) -> Optional[str]:
    t = _strip_code_fence(text)
    if not t:
        return None
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        if open_ch in t and close_ch in t:
            sub = t[t.find(open_ch) : t.rfind(close_ch) + 1]
            try:
                obj = json.loads(sub)
            except Exception:
                continue
            return json.dumps(obj, ensure_ascii=False)
    return None


def try_simple_compare_reply(
    user: str,
    prefer_it: bool = False,
    yesno_only: bool = False,
) -> Optional[str]:
    text = normalize(user).lower()

    def yn(ok: bool) -> str:
        if prefer_it:
            return "SI" if ok else "NO"
        return "YES" if ok else "NO"

    m_gt = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:greater than|larger than|bigger than|maggiore di|piu grande di)\s*(-?\d+(?:\.\d+)?)",
        text,
    )
    if m_gt:
        a = float(m_gt.group(1))
        b = float(m_gt.group(2))
        out = yn(a > b)
        if yesno_only:
            return out
        if out in {"YES", "SI"}:
            return "Yes."
        return "No."

    m_lt = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:less than|smaller than|minore di|piu piccolo di)\s*(-?\d+(?:\.\d+)?)",
        text,
    )
    if m_lt:
        a = float(m_lt.group(1))
        b = float(m_lt.group(2))
        out = yn(a < b)
        if yesno_only:
            return out
        if out in {"YES", "SI"}:
            return "Yes."
        return "No."

    m_sym = re.search(r"(-?\d+(?:\.\d+)?)\s*([<>])\s*(-?\d+(?:\.\d+)?)", text)
    if m_sym:
        a = float(m_sym.group(1))
        b = float(m_sym.group(3))
        op = m_sym.group(2)
        out = yn(a > b if op == ">" else a < b)
        if yesno_only:
            return out
        if out in {"YES", "SI"}:
            return "Yes."
        return "No."

    return None


def try_deterministic_yesno_reply(user: str, prefer_it: bool = False) -> Optional[str]:
    out = try_simple_compare_reply(user, prefer_it=prefer_it, yesno_only=True)
    if out is not None:
        return out
    out = _infer_yes_no_from_math(user, prefer_it=prefer_it)
    if out is not None:
        return out
    return None


def _infer_yes_no_from_math(user: str, prefer_it: bool = False) -> Optional[str]:
    text = normalize(user).lower()
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        a = float(m.group(1))
        op = m.group(2)
        b = float(m.group(3))
        c = float(m.group(4))
        try:
            if op == "+":
                val = a + b
            elif op == "-":
                val = a - b
            elif op == "*":
                val = a * b
            else:
                if abs(b) < 1e-12:
                    return None
                val = a / b
            ok = abs(val - c) < 1e-9
            if prefer_it:
                return "SI" if ok else "NO"
            return "YES" if ok else "NO"
        except Exception:
            return None

    m2 = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:greater than|larger than|maggiore di)\s*(-?\d+(?:\.\d+)?)", text)
    if m2:
        a = float(m2.group(1))
        b = float(m2.group(2))
        ok = a > b
        if prefer_it:
            return "SI" if ok else "NO"
        return "YES" if ok else "NO"

    m3 = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:less than|minore di)\s*(-?\d+(?:\.\d+)?)", text)
    if m3:
        a = float(m3.group(1))
        b = float(m3.group(2))
        ok = a < b
        if prefer_it:
            return "SI" if ok else "NO"
        return "YES" if ok else "NO"

    return None


def _is_summary_request(user: str) -> bool:
    low = normalize(user).lower()
    return low.startswith(
        (
            "summarize",
            "riassumi",
            "cosa dice questo articolo",
            "what does this article say",
        )
    )


def _detect_format_spec(user: str) -> Dict[str, bool]:
    low = normalize(user).lower()
    number_only = any(
        h in low
        for h in [
            "answer with a number only",
            "number only",
            "only a number",
            "numeric answer only",
            "solo un numero",
            "solo numero",
            "rispondi solo con un numero",
        ]
    )
    yes_no_only = any(
        h in low
        for h in [
            "reply yes or no only",
            "yes or no only",
            "answer yes or no only",
            "reply yes/no only",
            "reply with yes or no",
            "replay with yes or no",
            "yes or no",
            "rispondi si o no",
            "rispondi con si o no",
            "solo si o no",
            "si o no soltanto",
        ]
    )
    json_only = ("json" in low) and any(
        h in low
        for h in [
            "json only",
            "only json",
            "return json",
            "output json",
            "respond in json",
            "format json",
            "formato json",
            "solo json",
            "rispondi in json",
        ]
    )
    two_bullets = any(
        h in low
        for h in [
            "exactly 2 bullet points",
            "exactly two bullet points",
            "summarize as exactly 2 bullet points",
            "summarize as exactly two bullet points",
            "2 bullet points",
            "2 bullets",
            "due punti elenco",
            "2 punti elenco",
        ]
    )
    two_sentences = any(
        h in low
        for h in [
            "summarize in 2 sentences",
            "summarize in exactly 2 sentences",
            "summarize in exactly two sentences",
            "2 sentences",
            "two sentences",
            "riassumi in 2 frasi",
            "riassumi in due frasi",
            "2 frasi",
            "due frasi",
        ]
    )
    return {
        "number_only": number_only,
        "yes_no_only": yes_no_only,
        "json_only": json_only,
        "two_bullets": two_bullets,
        "two_sentences": two_sentences,
    }


def _prefer_italian_yes_no(user: str) -> bool:
    low = normalize(user).lower()
    return any(
        h in low
        for h in [
            "rispondi si o no",
            "solo si o no",
            "si o no soltanto",
        ]
    )


def _has_strict_format(spec: Dict[str, bool]) -> bool:
    return any(spec.values())


def _format_ok(spec: Dict[str, bool], answer: str) -> bool:
    if spec.get("number_only") and not _is_number_only(answer):
        return False
    if spec.get("yes_no_only") and not _is_yes_no_only(answer):
        return False
    if spec.get("json_only") and not _is_json_only(answer):
        return False
    if spec.get("two_bullets") and not _exactly_two_bullets(answer):
        return False
    if spec.get("two_sentences") and not _two_sentence_exact(answer):
        return False
    return True


def _apply_format_repair(user: str, answer: str) -> Tuple[str, Optional[str]]:
    spec = _detect_format_spec(user)
    if not _has_strict_format(spec):
        return answer, None
    if _format_ok(spec, answer):
        if spec.get("yes_no_only"):
            prefer_it = _prefer_italian_yes_no(user)
            deterministic = try_deterministic_yesno_reply(user, prefer_it=prefer_it)
            if deterministic is not None:
                return deterministic, "repair_yesno_deterministic"
        return answer, None

    if spec.get("number_only"):
        num = _extract_first_number(answer) or try_simple_math_reply(user)
        if num is not None:
            return num, "repair_number"
        return "0", "repair_number_default"

    if spec.get("yes_no_only"):
        prefer_it = _prefer_italian_yes_no(user)
        deterministic = try_deterministic_yesno_reply(user, prefer_it=prefer_it)
        if deterministic is not None:
            return deterministic, "repair_yesno_deterministic"
        yn = _extract_yes_no(answer)
        if yn is not None:
            if prefer_it:
                return ("SI" if yn in {"YES", "SI"} else "NO"), "repair_yesno"
            return ("YES" if yn in {"YES", "SI"} else "NO"), "repair_yesno"
        yn = _infer_yes_no_from_math(user, prefer_it=prefer_it)
        if yn is not None:
            return yn, "repair_yesno"
        return "NO", "repair_yesno_default"

    if spec.get("json_only"):
        blob = _extract_json_blob(answer)
        if blob is not None:
            return blob, "repair_json"
        return "{}", "repair_json_default"

    if spec.get("two_bullets") or spec.get("two_sentences"):
        if _is_summary_request(user):
            summ = try_extractive_summary_reply(user)
            if summ and _format_ok(spec, summ):
                return summ, "repair_extractive_summary"

        pair = _best_sentence_pair(answer)
        if pair:
            if spec.get("two_bullets"):
                fixed = f"- {pair[0]}\n- {pair[1]}"
            else:
                fixed = f"{pair[0]} {pair[1]}".strip()
            if _format_ok(spec, fixed):
                return fixed, "repair_from_answer"

        if spec.get("two_bullets"):
            return "- ...\n- ...", "repair_bullets_default"
        return "I don't know. I don't know.", "repair_sentences_default"

    return answer, None


def _needs_latest_web_search(user: str) -> bool:
    low = (user or "").lower()
    triggers = [
        "latest",
        "today",
        "news",
        "current",
        "recent",
        "ultime",
        "oggi",
        "aggiorn",
        "quali sono le ultime",
    ]
    return any(t in low for t in triggers)


def _prefer_general_web(user: str) -> bool:
    return False


def _allow_news_fallback(intent: str) -> bool:
    # Prefer focused pages first, but never leave the user with no answer.
    return True


def _entity_from_query(user: str) -> Optional[str]:
    low = normalize(user).lower().strip()
    pats = [
        r"^who is\s+(.+?)\??$",
        r"^what is\s+(.+?)\??$",
        r"^who was\s+(.+?)\??$",
        r"^tell me about\s+(.+?)\??$",
        r"^chi e\s+(.+?)\??$",
        r"^chi è\s+(.+?)\??$",
        r"^cos[ae] e\s+(.+?)\??$",
        r"^cos[ae] è\s+(.+?)\??$",
    ]
    for p in pats:
        m = re.match(p, low)
        if m:
            ent = m.group(1).strip(" .?!")
            if _looks_like_math_query(ent):
                return None
            if len(ent) >= 2 and "latest" not in ent and "ultim" not in ent:
                return ent
    return None


def _two_sentence_text(text: str) -> str:
    sents = _split_sentences(normalize(text))
    if not sents:
        return normalize(text)
    if len(sents) == 1:
        return sents[0]
    return f"{sents[0]} {sents[1]}"


def _wikipedia_summary(entity: str, timeout: int = 15) -> Optional[Dict[str, str]]:
    # Search best page title first.
    search_url = (
        "https://en.wikipedia.org/w/api.php?action=opensearch"
        f"&search={urllib.parse.quote_plus(entity)}&limit=1&namespace=0&format=json"
    )
    txt = _fetch_text(search_url, timeout=timeout)
    if not txt:
        return None
    try:
        data = json.loads(txt)
        titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    except Exception:
        return None
    if not titles:
        return None
    title = str(titles[0]).strip()
    if not title:
        return None

    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    s_txt = _fetch_text(summary_url, timeout=timeout)
    if not s_txt:
        return None
    try:
        s_data = json.loads(s_txt)
    except Exception:
        return None
    extract = normalize(s_data.get("extract", ""))
    page = normalize(
        (s_data.get("content_urls", {}).get("desktop", {}) or {}).get("page", "")
        or s_data.get("content_urls", {}).get("mobile", {}).get("page", "")
    )
    if not extract:
        return None
    return {"title": title, "extract": _two_sentence_text(extract), "url": page}


def _try_entity_answer(user: str, timeout: int = 15) -> Optional[Tuple[str, List[str]]]:
    ent = _entity_from_query(user)
    if not ent:
        return None
    w = _wikipedia_summary(ent, timeout=timeout)
    if not w:
        return None
    ans = w["extract"]
    src = [w["url"]] if w.get("url") else []
    if src:
        ans = f"{ans}\n\nSource:\n- {src[0]}"
    return ans, src


def _fuzzy_wiki_entity_candidates(entity: str) -> List[str]:
    e = normalize(entity).strip(" ,.")
    low = e.lower()
    out = [e]
    cleaned = low
    cleaned = re.sub(r"\b(architectures?|architecture|gpus?|cpus?)\b", "", cleaned).strip()
    if cleaned and cleaned != low:
        out.append(cleaned)

    uniq: List[str] = []
    seen = set()
    for x in out:
        k = x.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(x.strip())
    return uniq


def _wikipedia_summary_fuzzy(entity: str, timeout: int = 12) -> Optional[Dict[str, str]]:
    cands = _fuzzy_wiki_entity_candidates(entity)
    for i, cand in enumerate(cands):
        w = _wikipedia_summary(cand, timeout=timeout)
        if w is None:
            continue
        extract_low = normalize(w.get("extract", "")).lower()
        # Skip disambiguation-like pages when better candidates exist.
        if "may refer to" in extract_low and i < (len(cands) - 1):
            continue
        return w
    return None


def _clean_news_title(title: str) -> str:
    t = normalize(title)
    return re.sub(r"\s*-\s*[^-]+$", "", t).strip()


def _domain_from_url(url: str) -> str:
    try:
        net = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""
    if net.startswith("www."):
        net = net[4:]
    return net


def _official_only_requested(query: str) -> bool:
    low = normalize(query).lower()
    return any(
        x in low
        for x in (
            "official sources only",
            "official source only",
            "only official sources",
            "solo fonti ufficiali",
            "fonti ufficiali",
        )
    )


def _allowed_official_domains(query: str) -> List[str]:
    low = normalize(query).lower()
    toks = [t for t in re.findall(r"[a-z0-9]+", low) if len(t) >= 4 and t not in QUERY_STOPWORDS]
    generic = {
        "official",
        "sources",
        "source",
        "only",
        "latest",
        "recent",
        "current",
        "what",
        "which",
        "where",
        "when",
        "using",
        "with",
        "from",
    }
    out = [t for t in toks if t not in generic][:5]
    out.extend([".gov", ".edu"])
    # dedupe preserve order
    uniq: List[str] = []
    seen = set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _is_allowed_official_domain(url: str, allowed: List[str]) -> bool:
    d = _domain_from_url(url)
    if not d:
        return False
    for a in allowed:
        if a.startswith("."):
            if d.endswith(a):
                return True
        elif a in d:
            return True
    return False


def _decode_ddg_redirect(url: str) -> str:
    u = normalize(url)
    if not u:
        return ""
    if u.startswith("//"):
        u = "https:" + u
    if "duckduckgo.com/l/?" not in u:
        return u
    try:
        q = urllib.parse.urlparse(u).query
        params = urllib.parse.parse_qs(q)
        if "uddg" in params and params["uddg"]:
            return urllib.parse.unquote(params["uddg"][0])
    except Exception:
        pass
    return u


def _query_tokens(query: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", normalize(query).lower())
    return [t for t in toks if len(t) >= 2 and t not in QUERY_STOPWORDS]


def _trusted_domain_boost(url: str) -> int:
    d = _domain_from_url(url)
    if not d:
        return 0
    score = 0
    for hint in TRUSTED_DOMAIN_HINTS:
        if hint.startswith("."):
            if d.endswith(hint):
                score += 2
        elif hint in d:
            score += 3
    return score


def _search_duckduckgo(query: str, max_items: int = 8, timeout: int = 15) -> List[Dict[str, str]]:
    q = urllib.parse.quote_plus(normalize(query))
    url = f"https://duckduckgo.com/html/?q={q}"
    raw = _fetch_raw_html(url, timeout=timeout)
    if not raw:
        return []
    links = [
        (m.group(1), m.group(2))
        for m in re.finditer(
            r'<a rel=\"nofollow\" class=\"result__a\" href=\"([^\"]+)\">(.*?)</a>',
            raw,
            re.IGNORECASE | re.DOTALL,
        )
    ]
    snippets = [
        html.unescape(re.sub("<.*?>", "", m.group(1))).strip()
        for m in re.finditer(r'<a class=\"result__snippet\"[^>]*>(.*?)</a>', raw, re.IGNORECASE | re.DOTALL)
    ]
    out: List[Dict[str, str]] = []
    for i, (href, title_html) in enumerate(links[: max_items * 2]):
        title = html.unescape(re.sub("<.*?>", "", title_html)).strip()
        snip = snippets[i] if i < len(snippets) else ""
        link = _decode_ddg_redirect(href)
        if not title or not link:
            continue
        out.append({"title": normalize(title), "link": normalize(link), "snippet": normalize(snip), "pub_date": ""})
        if len(out) >= max_items:
            break
    return out


def _strip_tags(text: str) -> str:
    return normalize(html.unescape(re.sub(r"<.*?>", " ", text or "")))


def _search_bing_web(query: str, max_items: int = 8, timeout: int = 15) -> List[Dict[str, str]]:
    q = urllib.parse.quote_plus(normalize(query))
    url = f"https://www.bing.com/search?q={q}&setlang=en-US"
    raw = _fetch_raw_html(url, timeout=timeout)
    if not raw:
        return []
    out: List[Dict[str, str]] = []
    for m in re.finditer(r'<li class=\"b_algo\".*?</li>', raw, re.IGNORECASE | re.DOTALL):
        block = m.group(0)
        a = re.search(r'<h2>\s*<a href=\"([^\"]+)\"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
        if not a:
            continue
        link = normalize(a.group(1))
        title = _strip_tags(a.group(2))
        p = re.search(r"<p>(.*?)</p>", block, re.IGNORECASE | re.DOTALL)
        snip = _strip_tags(p.group(1)) if p else ""
        if not title or not link:
            continue
        out.append({"title": title, "link": link, "snippet": snip, "pub_date": ""})
        if len(out) >= max_items:
            break
    return out


def _web_hit_score(query: str, item: Dict[str, str]) -> int:
    text = normalize(f"{item.get('title', '')} {item.get('snippet', '')}").lower()
    q_toks = _query_tokens(query)
    overlap = sum(1 for t in q_toks if t in text)
    score = overlap * 3 + _trusted_domain_boost(item.get("link", ""))
    domain = _domain_from_url(item.get("link", ""))
    if any(h in domain for h in LOW_QUALITY_DOMAIN_HINTS):
        score -= 4

    ql = query.lower()
    if ("latest" in ql or "recent" in ql or "ultim" in ql) and ("latest" in text or "new" in text or "update" in text):
        score += 2
    if "compare" in ql and ("vs" in text or "versus" in text or "compare" in text or "difference" in text):
        score += 2
    return score


def _rank_web_hits(query: str, items: List[Dict[str, str]], max_items: int = 5) -> List[Dict[str, str]]:
    dedup: Dict[str, Dict[str, str]] = {}
    for it in items:
        key = normalize(it.get("link", "")).lower()
        if not key:
            key = normalize(it.get("title", "")).lower()
        if key and key not in dedup:
            dedup[key] = it
    ranked = list(dedup.values())
    ranked.sort(key=lambda it: _web_hit_score(query, it), reverse=True)
    return ranked[:max_items]


def _search_general_web(query: str, max_items: int = 5, timeout: int = 15) -> List[Dict[str, str]]:
    all_items: List[Dict[str, str]] = []
    for variant in _ddg_query_variants(query):
        all_items.extend(_search_duckduckgo(variant, max_items=max(6, max_items * 2), timeout=timeout))
    # DDG can be unstable; blend with Bing for better hit coverage.
    if len(all_items) < max_items:
        for variant in _ddg_query_variants(query):
            all_items.extend(_search_bing_web(variant, max_items=max(6, max_items * 2), timeout=timeout))
    return _rank_web_hits(query, all_items, max_items=max_items)


def _search_wikipedia_pages(query: str, max_items: int = 5, timeout: int = 12) -> List[Dict[str, str]]:
    url = (
        "https://en.wikipedia.org/w/api.php?action=opensearch"
        f"&search={urllib.parse.quote_plus(normalize(query))}&limit={max(1, int(max_items))}&namespace=0&format=json"
    )
    txt = _fetch_text(url, timeout=timeout)
    if not txt:
        return []
    try:
        data = json.loads(txt)
        titles = data[1] if isinstance(data, list) and len(data) > 1 else []
        descs = data[2] if isinstance(data, list) and len(data) > 2 else []
        links = data[3] if isinstance(data, list) and len(data) > 3 else []
    except Exception:
        return []

    out: List[Dict[str, str]] = []
    n = min(len(titles), len(links), max(1, int(max_items)))
    for i in range(n):
        title = normalize(str(titles[i]))
        link = normalize(str(links[i]))
        desc = normalize(str(descs[i] if i < len(descs) else ""))
        if not title or not link:
            continue
        if not desc:
            ws = _wikipedia_summary(title, timeout=timeout)
            if ws:
                desc = _split_sentences(ws.get("extract", ""))[0] if _split_sentences(ws.get("extract", "")) else ws.get("extract", "")
        out.append({"title": title, "link": link, "snippet": normalize(desc), "pub_date": ""})
    return out


def _latest_news_quality(query: str, items: List[Dict[str, str]]) -> int:
    if not items:
        return -999
    return max(_latest_title_score(it.get("title", ""), query) for it in items[:3])


def _clean_snippet(snippet: str, max_chars: int = 220) -> str:
    s = normalize(snippet)
    s = s.replace("|", " ")
    s = re.sub(r"\[\d+\]", " ", s)
    s = re.sub(r"\s*\|\s*[^|]{0,60}$", "", s)
    s = re.sub(r"\s+-\s+[^-]{0,60}$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[: max_chars - 3].rstrip() + "..."
    return s


def _numeric_signal_count(text: str) -> int:
    return len(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def _query_asks_numeric_fact(query: str) -> bool:
    ql = normalize(query).lower()
    return bool(
        re.search(r"\b(how many|how much|quanto|quanti|quanta)\b", ql)
        or re.search(r"\bwhat is\b.*\bof\b", ql)
        or ("vram" in ql)
        or ("memory" in ql)
        or ("ram" in ql)
    )


def _extract_factoid_answer(query: str, items: List[Dict[str, str]]) -> Optional[str]:
    q = normalize(query)
    if not items:
        return None

    corpus: List[str] = []
    for it in items[:4]:
        corpus.append(normalize(it.get("snippet", "")))
        corpus.append(normalize(it.get("title", "")))
    corpus = [c for c in corpus if c]
    if not corpus:
        return None

    # Generic factoid: choose informative snippet with strongest topical overlap.
    q_toks = _query_tokens(q)
    query_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", q.lower()))
    candidates: List[Tuple[int, int, int, int, str]] = []

    def add_candidate_text(txt: str) -> None:
        for s in _split_sentences(txt):
            s = normalize(s)
            if len(s) < 12:
                continue
            low_s = s.lower()
            overlap = sum(1 for t in q_toks if t in low_s)
            if overlap <= 0:
                continue
            nums = re.findall(r"\b\d+(?:\.\d+)?\b", low_s)
            numeric = len(nums)
            novel_numeric = sum(1 for n in nums if n not in query_nums)
            score = overlap * 4 + min(4, numeric * 2) + min(6, novel_numeric * 3)
            if "|" in s:
                score -= 3
            if re.search(r"\[\d+\]", s):
                score -= 2
            if re.search(r"\b(toggle|subsection|contents?)\b", low_s):
                score -= 4
            if re.search(r"\b\d+\.\d+\b", low_s):
                score -= 2
            has_predicate = any(v in low_s for v in (" is ", " are ", " was ", " were ", " has ", " have ", " include", " includes ", " e ", " sono ", " ha "))
            if (not has_predicate) and numeric == 0:
                score -= 2
            candidates.append((score, novel_numeric, numeric, len(s), s))

    for c in corpus:
        add_candidate_text(c)
        low_c = c.lower()
        for tok in q_toks[:8]:
            start = 0
            hits = 0
            while hits < 3:
                pos = low_c.find(tok, start)
                if pos < 0:
                    break
                left = max(0, pos - 120)
                right = min(len(c), pos + len(tok) + 120)
                add_candidate_text(c[left:right])
                start = pos + len(tok)
                hits += 1
    if not candidates:
        return None

    ask_numeric = _query_asks_numeric_fact(q)
    if ask_numeric:
        numeric_candidates = [c for c in candidates if c[2] > 0]
        if numeric_candidates:
            candidates = sorted(numeric_candidates, key=lambda x: (-x[1], -x[0], -x[2], x[3]))
        else:
            candidates = sorted(candidates, key=lambda x: (-x[0], -x[1], -x[2], x[3]))
    else:
        candidates = sorted(candidates, key=lambda x: (-x[0], -x[1], -x[2], x[3]))

    def is_noisy_chunk(txt: str) -> bool:
        low_txt = txt.lower()
        if any(x in low_txt for x in ("---", ".jpg", ".png", ".svg", ".gif")):
            return True
        if any(x in low_txt for x in ("jump to content", "main menu", "privacy policy", "contact us", "sign in")):
            return True
        noise_ratio = sum(1 for ch in txt if ch in "-|[]{}") / max(1, len(txt))
        return noise_ratio > 0.08

    best = ""
    for _, _, _, _, raw in candidates:
        cleaned = _clean_snippet(raw, max_chars=180)
        if cleaned and not is_noisy_chunk(cleaned):
            best = cleaned
            break
    if not best:
        return None
    if best and best[-1] not in ".!?":
        best += "."
    return best or None


def _intent_answer_quality_ok(query: str, answer: str) -> bool:
    low = normalize(answer).lower()
    if not low:
        return False
    # Generic quality: at least one topical token overlap or numeric signal for factoid questions.
    q_toks = _query_tokens(query)
    overlap = sum(1 for t in q_toks if t in low)
    has_numeric = _numeric_signal_count(low) >= 1
    if _query_asks_numeric_fact(query):
        return has_numeric and overlap >= 1
    if len(q_toks) >= 4:
        return overlap >= 2 or (overlap >= 1 and has_numeric)
    return overlap >= 1 or has_numeric


def _format_general_web_answer(query: str, items: List[Dict[str, str]]) -> str:
    if not items:
        return "I couldn't find reliable web results for that query right now."
    fact = _extract_factoid_answer(query, items)
    if fact:
        src = "\n".join(f"- {it['link']}" for it in items[:3] if it.get("link"))
        return f"{fact}\n\nSources:\n{src}"
    lead = items[0]
    second = items[1] if len(items) > 1 else lead
    t1 = normalize(lead.get("title", ""))
    t2 = normalize(second.get("title", ""))

    s1 = _clean_snippet(lead.get("snippet", "") or lead.get("title", ""))
    s2 = _clean_snippet(second.get("snippet", "") or second.get("title", ""))

    resp = (
        f"Web summary:\n"
        f"- {t1}: {s1}\n"
        f"- {t2}: {s2}"
    )
    src = "\n".join(f"- {it['link']}" for it in items[:3] if it.get("link"))
    return f"{resp}\n\nSources:\n{src}"


def _web_no_result_answer(query: str) -> str:
    return (
        "I couldn't find reliable web results for that query right now. "
        "Please try a more specific version (topic + year/source)."
    )


def _intent_retry_query(query: str) -> str:
    return f"{normalize(query)} reliable primary sources".strip()


def _try_general_web_answer(query: str, max_items: int = 5, timeout: int = 15) -> Optional[Tuple[str, List[str], str]]:
    q = normalize(query)
    official_only = _official_only_requested(q)

    base_items = _search_general_web(q, max_items=max_items, timeout=timeout)
    ddg_items = list(base_items)
    if official_only:
        allowed = _allowed_official_domains(q)
        ddg_items = [it for it in ddg_items if _is_allowed_official_domain(it.get("link", ""), allowed)]
        if not ddg_items:
            official_q = f"{q} official site documentation specs"
            ddg_items = _search_general_web(official_q, max_items=max_items, timeout=timeout)
            ddg_items = [it for it in ddg_items if _is_allowed_official_domain(it.get("link", ""), allowed)]
        if not ddg_items:
            seed_domains: List[str] = []
            content_tokens = [
                t for t in _query_tokens(q)
                if t not in {"official", "sources", "source", "only", "use"}
            ]
            content_terms = " ".join(content_tokens[:6]) or q
            for tok in _query_tokens(q):
                if len(tok) < 4:
                    continue
                for tld in (".com", ".org", ".gov", ".edu"):
                    seed_domains.append(tok + tld)
            uniq_domains: List[str] = []
            seen_domains = set()
            for d in seed_domains:
                if d in seen_domains:
                    continue
                seen_domains.add(d)
                uniq_domains.append(d)
            site_hits: List[Dict[str, str]] = []
            for dom in uniq_domains[:6]:
                site_q = f"site:{dom} {content_terms}"
                for it in _search_general_web(site_q, max_items=max_items, timeout=timeout):
                    if _domain_from_url(it.get("link", "")).endswith(dom):
                        site_hits.append(it)
            ddg_items = _rank_web_hits(q, site_hits, max_items=max_items)
        elif len(ddg_items) < max(2, max_items // 2):
            official_q = f"{q} official technical specifications"
            extra = _search_general_web(official_q, max_items=max_items, timeout=timeout)
            extra = [it for it in extra if _is_allowed_official_domain(it.get("link", ""), allowed)]
            merged = ddg_items + extra
            ddg_items = _rank_web_hits(q, merged, max_items=max_items)
    if ddg_items:
        ans = _format_general_web_answer(q, ddg_items)
        if (not _intent_answer_quality_ok(q, ans)) or _looks_generic_or_bad_answer(ans):
            retry_q = _intent_retry_query(q)
            if official_only:
                retry_q = f"{q} official technical specifications detailed"
                allowed = _allowed_official_domains(q)
                retry_items = _search_general_web(retry_q, max_items=max_items, timeout=timeout)
                retry_items = [it for it in retry_items if _is_allowed_official_domain(it.get("link", ""), allowed)]
            else:
                retry_items = _search_general_web(retry_q, max_items=max_items, timeout=timeout)
            if retry_items:
                retry_ans = _format_general_web_answer(q, retry_items)
                if _intent_answer_quality_ok(q, retry_ans) and (not _looks_generic_or_bad_answer(retry_ans)):
                    return retry_ans, [it["link"] for it in retry_items[:3] if it.get("link")], "web_general_retry"
        return ans, [it["link"] for it in ddg_items[:3] if it.get("link")], "web_general"

    wiki_items = _rank_web_hits(
        q,
        _search_wikipedia_pages(q, max_items=max_items, timeout=min(timeout, 12)),
        max_items=max_items,
    )
    if official_only:
        wiki_items = []
    if wiki_items:
        ans = _format_general_web_answer(q, wiki_items)
        if _intent_answer_quality_ok(q, ans) and (not _looks_generic_or_bad_answer(ans)):
            return ans, [it["link"] for it in wiki_items[:3] if it.get("link")], "web_wikipedia_search"

    if (not official_only) and _allow_news_fallback("general"):
        news_items = _search_google_news(q, max_items=max_items, timeout=timeout)
        if news_items:
            ans = _format_latest_answer(q, news_items)
            return ans, [it["link"] for it in news_items[:3] if it.get("link")], "web_general_news_fallback"

    if official_only:
        fallback_items = list(base_items)
        return (
            (
                "I couldn't verify enough official sources for that query right now. "
                "Using the best available sources instead.\n\n"
                + _format_general_web_answer(q, fallback_items)
            )
            if fallback_items
            else "I couldn't find enough official sources for that query right now.",
            [it["link"] for it in fallback_items[:3] if it.get("link")] if fallback_items else [],
            "web_official_relaxed" if fallback_items else "web_official_no_results",
        )

    return None


def _needs_general_web_search(user: str) -> bool:
    low = normalize(user).lower()
    if not low or _looks_like_math_query(low):
        return False
    if low.startswith(("summarize", "riassumi", "cosa dice questo articolo", "what does this article say")):
        return False
    if "answer with a number only" in low or "reply yes or no only" in low:
        return False
    if _needs_latest_web_search(low):
        return True
    return ("?" in low) and len(_query_tokens(low)) >= 2


def _ddg_query_variants(query: str) -> List[str]:
    q = normalize(query)
    return [q] if q else []


def _latest_query_variants(query: str) -> List[str]:
    q = normalize(query)
    return [q] if q else []


def _parse_pub_ts(pub_date: str) -> float:
    try:
        return email.utils.parsedate_to_datetime(pub_date).timestamp()
    except Exception:
        return 0.0


def _latest_title_score(title: str, query: str) -> int:
    t = title.lower()
    q_toks = _query_tokens(query)
    score = sum(2 for tok in q_toks if tok in t)
    if any(k in t for k in ("latest", "new", "update", "announc", "launch", "release")):
        score += 2
    if any(k in t for k in ("opinion", "rumor", "forum")):
        score -= 1
    return score


def _search_google_news(query: str, max_items: int = 5, timeout: int = 15) -> List[Dict[str, str]]:
    all_items: List[Dict[str, str]] = []
    for variant in _latest_query_variants(query):
        q = urllib.parse.quote_plus(variant)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        txt = _fetch_text(url, timeout=timeout)
        if not txt:
            continue
        try:
            root = ET.fromstring(txt)
        except Exception:
            continue
        for item in root.findall("./channel/item"):
            title = _clean_news_title(item.findtext("title", default=""))
            link = normalize(item.findtext("link", default=""))
            pub = normalize(item.findtext("pubDate", default=""))
            desc = _strip_tags(item.findtext("description", default=""))
            source_el = item.find("source")
            source_name = normalize(source_el.text if source_el is not None else "")
            source_url = normalize(source_el.get("url", "")) if source_el is not None else ""
            if title and link:
                all_items.append(
                    {
                        "title": title,
                        "link": link,
                        "pub_date": pub,
                        "snippet": desc,
                        "source_name": source_name,
                        "source_url": source_url,
                    }
                )

    # dedupe by link/title, then rank by relevance + recency
    dedup: Dict[str, Dict[str, str]] = {}
    for it in all_items:
        key = (it["link"] or it["title"]).strip().lower()
        if key and key not in dedup:
            dedup[key] = it
    ranked = list(dedup.values())
    ranked.sort(
        key=lambda it: (_latest_title_score(it["title"], query), _parse_pub_ts(it.get("pub_date", ""))),
        reverse=True,
    )
    return ranked[:max_items]


def _format_latest_answer(query: str, items: List[Dict[str, str]]) -> str:
    if not items:
        return "I couldn't find recent sources for that query right now."
    lead = items[0]
    title = _clean_news_title(lead.get("title", "")).rstrip(".")
    snip = _clean_snippet(lead.get("snippet", "") or title, max_chars=180)
    resp = f"Latest reported update: {title}."
    if snip and snip.lower() not in title.lower():
        resp += f" {snip}"

    links = []
    for it in items[:3]:
        candidate = normalize(it.get("source_url", "")) or normalize(it.get("link", ""))
        if candidate and candidate not in links:
            links.append(candidate)
    src = "\n".join(f"- {u}" for u in links)
    return f"{resp}\n\nSources:\n{src}"


def _append_web_log(path: str, record: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_turn_log(
    path: str,
    user: str,
    answer: str,
    mode: str,
    source_urls: Optional[List[str]] = None,
    meta: Optional[Dict[str, object]] = None,
) -> None:
    if not path:
        return
    rec: Dict[str, object] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "user": normalize(user),
        "answer": normalize(answer),
    }
    if source_urls:
        rec["source_urls"] = [normalize(u) for u in source_urls[:3] if normalize(u)]
    if meta:
        rec.update(meta)
    _append_web_log(path, rec)


def _looks_like_math_query(user: str) -> bool:
    u = (user or "").lower()
    if re.search(r"\d+\s*[+\-*/]\s*\d+", u):
        return True
    if re.search(r"(-?\d+(?:\.\d+)?)\s*(?:greater than|larger than|bigger than|less than|smaller than|minore di|maggiore di)\s*(-?\d+(?:\.\d+)?)", u):
        return True
    if re.search(r"-?\d+(?:\.\d+)?\s*[<>]\s*-?\d+(?:\.\d+)?", u):
        return True
    return False


def _looks_like_factual_query(user: str) -> bool:
    low = normalize(user).lower()
    if not low:
        return False
    if low.startswith(("summarize", "riassumi", "cosa dice questo articolo", "what does this article say")):
        return False
    if _looks_like_math_query(low):
        return False
    if any(s in low for s in ("what can you do", "chi sei", "who are you")):
        return False
    if any(t in low for t in ("latest", "today", "news", "current", "recent", "ultime", "oggi", "aggiorn")):
        return True
    starts = (
        "what is", "who is", "when is", "when did", "where is", "which", "quali", "chi e",
        "cosa e", "dove", "quanto", "how many", "how much", "tell me about",
    )
    return ("?" in low) and low.startswith(starts)


def _looks_generic_or_bad_answer(ans: str) -> bool:
    low = normalize(ans).lower()
    if not low:
        return True
    if any(h in low for h in GENERIC_NONANSWER_HINTS):
        return True
    if low.count("-") >= 20:
        return True
    words = re.findall(r"[a-z0-9']+", low)
    if len(words) >= 12:
        uniq = len(set(words)) / max(1, len(words))
        if uniq < 0.45:
            return True
    if re.search(r"\b(\w+)(?:\s+\1){2,}\b", low):
        return True
    return False


def _should_confidence_web_fallback(
    user: str,
    ans: str,
    avg_token_prob: float,
    threshold: float,
) -> bool:
    if not _looks_like_factual_query(user):
        return False
    if len((ans or "").strip()) < 18:
        return True
    if _looks_generic_or_bad_answer(ans):
        return True
    return avg_token_prob < threshold


def _format_number(n: float) -> str:
    if abs(n - round(n)) < 1e-9:
        return str(int(round(n)))
    s = f"{n:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def try_simple_math_reply(user: str) -> Optional[str]:
    text = normalize(user).lower()
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    a = float(m.group(1))
    op = m.group(2)
    b = float(m.group(3))
    try:
        if op == "+":
            out = a + b
        elif op == "-":
            out = a - b
        elif op == "*":
            out = a * b
        else:
            if abs(b) < 1e-12:
                return "undefined"
            out = a / b
    except Exception:
        return None
    return _format_number(out)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack((out1, out2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=q.dtype)
        q = self.rope.apply_rotary(q, cos, sin)
        k = self.rope.apply_rotary(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        hidden = int((dim * 8) / 3)
        hidden = (hidden + 255) // 256 * 256
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FastGPT(nn.Module):
    def __init__(self, vocab, dim, heads, layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, dim)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight
        print(f"Model params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        if x.shape[1] > self.block_size:
            x = x[:, -self.block_size:]
        h = self.drop(self.tok(x))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm_f(h))


# LoRA (same injection scheme as training script)
LORA_R = 16
LORA_ALPHA = 32
LORA_SCALE = LORA_ALPHA / LORA_R
LORA_TARGET = ["attn.qkv", "attn.proj"]


def inject_lora(model: nn.Module, device: str):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in LORA_TARGET):
            in_f, out_f = module.in_features, module.out_features
            module.lora_A = nn.Parameter(torch.zeros(LORA_R, in_f, device=device))
            module.lora_B = nn.Parameter(torch.zeros(out_f, LORA_R, device=device))
            module.lora_scale = float(LORA_SCALE)
            nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B)

            orig_forward = module.forward

            def forward(x, orig_forward=orig_forward, m=module):
                return orig_forward(x) + ((x @ m.lora_A.T) @ m.lora_B.T) * m.lora_scale

            module.forward = forward


def load_lora_state_dict(model: nn.Module, lora_sd: Dict[str, torch.Tensor], device: str):
    for name, module in model.named_modules():
        keyA = f"{name}.lora_A"
        keyB = f"{name}.lora_B"
        keyS = f"{name}.lora_scale"
        if keyA in lora_sd and hasattr(module, "lora_A"):
            module.lora_A.data.copy_(lora_sd[keyA].to(device))
            module.lora_B.data.copy_(lora_sd[keyB].to(device))
            module.lora_scale = float(lora_sd.get(keyS, module.lora_scale))


@torch.no_grad()
def sample_top_p(probs: torch.Tensor, top_p: float) -> int:
    if top_p >= 1.0:
        # No nucleus filter
        return int(torch.argmax(probs).item())

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
def generate(
    model,
    sp,
    device,
    prompt: str,
    temperature: float,
    top_p: float,
    return_meta: bool = False,
) -> str | Tuple[str, Dict[str, float]]:
    model.eval()
    ids = sp.encode(normalize(prompt), out_type=int)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    eos = sp.eos_id()
    chosen_probs: List[float] = []

    for _ in range(MAX_NEW_TOKENS):
        logits = model(x[:, -BLOCK_SIZE:])[:, -1, :]
        t = float(temperature)
        p = float(top_p)
        if t > 0:
            logits = logits / max(1e-6, t)
        probs = F.softmax(logits, dim=-1).squeeze(0)

        # Evaluation-first decoding:
        # - greedy if temperature<=0 or top_p>=1
        # - nucleus sampling only when explicitly requested (top_p<1)
        if t <= 0.0 or p >= 1.0:
            nxt = int(torch.argmax(probs).item())
        else:
            nxt = sample_top_p(probs, p)
        chosen_probs.append(float(probs[nxt].item()))

        x = torch.cat([x, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        if eos is not None and nxt == eos:
            break

    text = sp.decode(x[0].tolist())
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1]
    ans = text.strip()
    if not return_meta:
        return ans
    first = chosen_probs[:40] if chosen_probs else [0.0]
    avg_prob = float(sum(first) / max(1, len(first)))
    meta = {"avg_token_prob": avg_prob, "generated_tokens": float(len(chosen_probs))}
    return ans, meta


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lora_adapter", default="")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--no_web", action="store_true", help="Disable URL/latest web routing")
    ap.add_argument("--web_timeout", type=int, default=20)
    ap.add_argument("--web_results", type=int, default=6)
    ap.add_argument("--web_log", default="data/web_chat_log.jsonl")
    ap.add_argument(
        "--turn_log",
        default="data/chat_turns_log.jsonl",
        help="Full chat-turn log used for daily micro-retrain data generation",
    )
    ap.add_argument(
        "--no_confidence_web",
        action="store_true",
        help="Disable confidence-triggered web fallback for factual questions",
    )
    ap.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.22,
        help="Fallback to web when avg token confidence is below this threshold",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab = sp.get_piece_size()

    model = FastGPT(vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(device)

    ckpt = torch.load(args.base_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    # Only inject LoRA if we are actually going to load it
    if args.lora_adapter:
        inject_lora(model, device)
        lora_sd = torch.load(args.lora_adapter, map_location="cpu")
        load_lora_state_dict(model, lora_sd, device)
        print(f"Loaded LoRA: {args.lora_adapter}")

    print("\nType messages.")
    print("Summarization:  Summarize: <text>")
    print("Commands: /exit\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() in ("/exit", "/quit"):
            break

        format_spec = _detect_format_spec(user)
        strict_format = _has_strict_format(format_spec)

        if not args.no_web:
            url = _find_first_url(user)
            if url and (not strict_format or _is_summary_request(user)):
                article = _fetch_article_text(url, timeout=args.web_timeout)
                if article:
                    summ = _summarize_plain_text(article, bullets=bool(format_spec.get("two_bullets")))
                    if summ:
                        ans = summ if strict_format else f"{summ}\n\nSource:\n- {url}"
                        print(f"Bot: {ans}\n")
                        _append_turn_log(
                            args.turn_log,
                            user=user,
                            answer=ans,
                            mode="url",
                            source_urls=[url],
                        )
                        _append_web_log(
                            args.web_log,
                            {
                                "ts_utc": datetime.now(timezone.utc).isoformat(),
                                "mode": "url",
                                "query": user,
                                "source_urls": [url],
                                "answer": ans,
                            },
                        )
                        continue

            if strict_format:
                # Skip web routing for strict-format requests to avoid source blocks.
                pass
            elif _needs_latest_web_search(user) and not _prefer_general_web(user):
                items = _search_google_news(user, max_items=max(1, args.web_results), timeout=args.web_timeout)
                q_score = _latest_news_quality(user, items)
                if q_score < 3:
                    # Fallback to broader web search when RSS news is off-topic/noisy.
                    web_try = _try_general_web_answer(
                        user,
                        max_items=max(1, args.web_results),
                        timeout=args.web_timeout,
                    )
                    if web_try is not None:
                        ans, src_urls, mode = web_try
                        print(f"Bot: {ans}\n")
                        _append_turn_log(
                            args.turn_log,
                            user=user,
                            answer=ans,
                            mode=mode if mode != "web_general" else "latest_ddg_fallback",
                            source_urls=src_urls,
                        )
                        _append_web_log(
                            args.web_log,
                            {
                                "ts_utc": datetime.now(timezone.utc).isoformat(),
                                "mode": mode if mode != "web_general" else "latest_ddg_fallback",
                                "query": user,
                                "source_urls": src_urls,
                                "answer": ans,
                            },
                        )
                        continue

                if items:
                    ans = _format_latest_answer(user, items)
                    print(f"Bot: {ans}\n")
                    _append_turn_log(
                        args.turn_log,
                        user=user,
                        answer=ans,
                        mode="latest",
                        source_urls=[it["link"] for it in items[:3]],
                    )
                    _append_web_log(
                        args.web_log,
                        {
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "mode": "latest",
                            "query": user,
                            "source_urls": [it["link"] for it in items[:3]],
                            "answer": ans,
                        },
                        )
                    continue

            entity_ans = _try_entity_answer(user, timeout=args.web_timeout)
            if entity_ans is not None:
                ans, src_urls = entity_ans
                print(f"Bot: {ans}\n")
                _append_turn_log(
                    args.turn_log,
                    user=user,
                    answer=ans,
                    mode="entity",
                    source_urls=src_urls[:3],
                )
                _append_web_log(
                    args.web_log,
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "mode": "entity",
                        "query": user,
                        "source_urls": src_urls[:3],
                        "answer": ans,
                    },
                )
                continue

            if strict_format:
                # Skip web routing for strict-format requests to avoid source blocks.
                pass
            elif _needs_general_web_search(user):
                web_try = _try_general_web_answer(
                    user,
                    max_items=max(1, args.web_results),
                    timeout=args.web_timeout,
                )
                if web_try is not None:
                    ans, src_urls, mode = web_try
                    print(f"Bot: {ans}\n")
                    _append_turn_log(
                        args.turn_log,
                        user=user,
                        answer=ans,
                        mode=mode,
                        source_urls=src_urls,
                    )
                    _append_web_log(
                        args.web_log,
                        {
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "mode": mode,
                            "query": user,
                            "source_urls": src_urls,
                            "answer": ans,
                        },
                    )
                    continue

                ans = _web_no_result_answer(user)
                print(f"Bot: {ans}\n")
                _append_turn_log(
                    args.turn_log,
                    user=user,
                    answer=ans,
                    mode="web_no_results",
                )
                _append_web_log(
                    args.web_log,
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "mode": "web_no_results",
                        "query": user,
                        "answer": ans,
                    },
                )
                continue

        summary_fallback = try_extractive_summary_reply(user)
        if summary_fallback is not None:
            print(f"Bot: {summary_fallback}\n")
            _append_turn_log(
                args.turn_log,
                user=user,
                answer=summary_fallback,
                mode="summary_fallback",
            )
            continue

        math_fallback = try_simple_math_reply(user)
        if math_fallback is not None:
            print(f"Bot: {math_fallback}\n")
            _append_turn_log(
                args.turn_log,
                user=user,
                answer=math_fallback,
                mode="math_fallback",
            )
            continue

        user_norm = canonicalize_user_message(user)
        prompt = f"User: {user_norm}\nAssistant:"
        ans, meta = generate(
            model,
            sp,
            device,
            prompt,
            args.temperature,
            args.top_p,
            return_meta=True,
        )
        raw_ans = ans
        avg_prob = float(meta.get("avg_token_prob", 0.0))
        repaired_ans, repair_mode = _apply_format_repair(user, ans)
        if repair_mode:
            ans = repaired_ans

        if (not args.no_web) and (not args.no_confidence_web) and (not strict_format):
            if _should_confidence_web_fallback(user, ans, avg_prob, float(args.confidence_threshold)):
                entity_ans = _try_entity_answer(user, timeout=args.web_timeout)
                if entity_ans is not None:
                    web_ans, src_urls = entity_ans
                    print(f"Bot: {web_ans}\n")
                    _append_turn_log(
                        args.turn_log,
                        user=user,
                        answer=web_ans,
                        mode="confidence_entity_fallback",
                        source_urls=src_urls[:3],
                        meta={
                            "model_answer": normalize(ans),
                            "model_avg_token_prob": round(avg_prob, 4),
                        },
                    )
                    _append_web_log(
                        args.web_log,
                        {
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "mode": "confidence_entity_fallback",
                            "query": user,
                            "model_answer": ans,
                            "model_avg_token_prob": round(avg_prob, 4),
                            "source_urls": src_urls[:3],
                            "answer": web_ans,
                        },
                    )
                    continue

                if _needs_latest_web_search(user) and not _prefer_general_web(user):
                    items = _search_google_news(
                        user,
                        max_items=max(1, args.web_results),
                        timeout=args.web_timeout,
                    )
                    if items:
                        web_ans = _format_latest_answer(user, items)
                        print(f"Bot: {web_ans}\n")
                        _append_turn_log(
                            args.turn_log,
                            user=user,
                            answer=web_ans,
                            mode="confidence_latest_fallback",
                            source_urls=[it["link"] for it in items[:3]],
                            meta={
                                "model_answer": normalize(ans),
                                "model_avg_token_prob": round(avg_prob, 4),
                            },
                        )
                        _append_web_log(
                            args.web_log,
                            {
                                "ts_utc": datetime.now(timezone.utc).isoformat(),
                                "mode": "confidence_latest_fallback",
                                "query": user,
                                "model_answer": ans,
                                "model_avg_token_prob": round(avg_prob, 4),
                                "source_urls": [it["link"] for it in items[:3]],
                                "answer": web_ans,
                            },
                        )
                        continue

                if _needs_general_web_search(user):
                    web_try = _try_general_web_answer(
                        user,
                        max_items=max(1, args.web_results),
                        timeout=args.web_timeout,
                    )
                    if web_try is not None:
                        web_ans, src_urls, mode = web_try
                        print(f"Bot: {web_ans}\n")
                        _append_turn_log(
                            args.turn_log,
                            user=user,
                            answer=web_ans,
                            mode=f"confidence_{mode}",
                            source_urls=src_urls,
                            meta={
                                "model_answer": normalize(ans),
                                "model_avg_token_prob": round(avg_prob, 4),
                            },
                        )
                        _append_web_log(
                            args.web_log,
                            {
                                "ts_utc": datetime.now(timezone.utc).isoformat(),
                                "mode": f"confidence_{mode}",
                                "query": user,
                                "model_answer": ans,
                                "model_avg_token_prob": round(avg_prob, 4),
                                "source_urls": src_urls,
                                "answer": web_ans,
                            },
                        )
                        continue

                if _needs_general_web_search(user) or _needs_latest_web_search(user):
                    web_ans = _web_no_result_answer(user)
                    print(f"Bot: {web_ans}\n")
                    _append_turn_log(
                        args.turn_log,
                        user=user,
                        answer=web_ans,
                        mode="confidence_web_no_results",
                        meta={
                            "model_answer": normalize(ans),
                            "model_avg_token_prob": round(avg_prob, 4),
                        },
                    )
                    _append_web_log(
                        args.web_log,
                        {
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "mode": "confidence_web_no_results",
                            "query": user,
                            "model_answer": ans,
                            "model_avg_token_prob": round(avg_prob, 4),
                            "answer": web_ans,
                        },
                    )
                    continue

        print(f"Bot: {ans}\n")
        _append_turn_log(
            args.turn_log,
            user=user,
            answer=ans,
            mode="model_repair" if repair_mode else "model",
            meta={
                "model_avg_token_prob": round(avg_prob, 4),
                "repair_mode": repair_mode,
                "model_answer": normalize(raw_ans) if repair_mode else None,
            },
        )


if __name__ == "__main__":
    main()
