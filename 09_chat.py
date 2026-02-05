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
    "nvidia.com",
    "amd.com",
    "intel.com",
    "spacex.com",
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

    # Match "Summarize in 2 sentences:" prefix
    if low.startswith("summarize in 2 sentences:"):
        body = u.split(":", 1)[1].strip()
        return "Summarize in 2 sentences:\n" + body

    # Match "Summarize:" prefix
    if low.startswith("summarize:"):
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
    if low.startswith("summarize as exactly 2 bullet points:"):
        mode = "bullet"
        body = u.split(":", 1)[1].strip()
    elif low.startswith("summarize in 2 sentences:"):
        mode = "two"
        body = u.split(":", 1)[1].strip()
    elif low.startswith("summarize:"):
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


def _query_intent(user: str) -> str:
    low = normalize(user).lower()
    if any(k in low for k in ("tourist attractions", "travel tips", "visit tokyo", "visiting tokyo", "london")):
        return "travel"
    if "spacex" in low and any(k in low for k in ("launch", "outcome", "mission")):
        return "launches"
    if "regulation" in low and "ai" in low and any(k in low for k in ("eu", "europe", "european")):
        return "regulation"
    if any(k in low for k in ("stock", "inflation", "market", "economy", "economic", "investments")):
        return "economy"
    if any(k in low for k in ("compare", "vs", "versus", "differ", "difference")):
        return "compare"
    if any(k in low for k in ("spec", "key features", "feature")) and any(k in low for k in ("gpu", "cpu", "intel", "nvidia", "amd")):
        return "specs"
    return "general"


def _prefer_general_web(user: str) -> bool:
    intent = _query_intent(user)
    return intent in {"specs", "compare", "launches", "travel", "regulation", "economy"}


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


def _web_hit_score(query: str, item: Dict[str, str]) -> int:
    text = normalize(f"{item.get('title', '')} {item.get('snippet', '')}").lower()
    q_toks = _query_tokens(query)
    overlap = sum(1 for t in q_toks if t in text)
    score = overlap * 3 + _trusted_domain_boost(item.get("link", ""))
    domain = _domain_from_url(item.get("link", ""))
    if any(h in domain for h in LOW_QUALITY_DOMAIN_HINTS):
        score -= 4

    ql = query.lower()
    intent = _query_intent(query)
    if ("latest" in ql or "recent" in ql or "ultim" in ql) and ("launch" in text or "announc" in text or "new" in text):
        score += 2
    if "gpu" in ql and ("rtx" in text or "geforce" in text):
        score += 3
    if "spec" in ql and ("spec" in text or "gb" in text or "memory" in text or "cuda" in text):
        score += 2
    if "compare" in ql and ("vs" in text or "versus" in text or "compare" in text):
        score += 2
    if intent == "launches" and ("launch" in text or "mission" in text or "outcome" in text or "land" in text):
        score += 3
    if intent == "regulation" and ("act" in text or "regulation" in text or "compliance" in text):
        score += 2
    if intent == "economy" and ("inflation" in text or "market" in text or "stock" in text):
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
    return _rank_web_hits(query, all_items, max_items=max_items)


def _latest_news_quality(query: str, items: List[Dict[str, str]]) -> int:
    if not items:
        return -999
    return max(_latest_title_score(it.get("title", ""), query) for it in items[:3])


def _format_general_web_answer(query: str, items: List[Dict[str, str]]) -> str:
    if not items:
        return "I couldn't find reliable web results for that query right now."
    intent = _query_intent(query)
    lead = items[0]
    second = items[1] if len(items) > 1 else lead
    s1 = lead.get("snippet", "") or lead.get("title", "")
    s2 = second.get("snippet", "") or second.get("title", "")
    s1 = _two_sentence_text(s1)
    s2 = _two_sentence_text(s2)
    if intent == "specs":
        resp = f"Key specs/features from current sources: {s1} Another source adds: {s2}"
    elif intent == "compare":
        resp = f"Comparison highlights from web sources: {s1} Another source notes: {s2}"
    elif intent == "launches":
        resp = f"Recent launch/outcome sources report: {s1} Another source reports: {s2}"
    elif intent == "travel":
        resp = f"Current travel guidance says: {s1} Another source suggests: {s2}"
    elif intent == "regulation":
        resp = f"Recent regulatory sources indicate: {s1} Another source states: {s2}"
    elif intent == "economy":
        resp = f"Recent market/economic sources indicate: {s1} Another source reports: {s2}"
    else:
        resp = f"Web results indicate: {s1} Another source reports: {s2}"
    src = "\n".join(f"- {it['link']}" for it in items[:3] if it.get("link"))
    return f"{resp}\n\nSources:\n{src}"


def _web_no_result_answer(query: str) -> str:
    return (
        "I couldn't find reliable web results for that query right now. "
        "Please try a more specific version (topic + year/source)."
    )


def _needs_general_web_search(user: str) -> bool:
    low = normalize(user).lower()
    if not low or _looks_like_math_query(low):
        return False
    if low.startswith(("summarize", "riassumi", "cosa dice questo articolo", "what does this article say")):
        return False
    if "answer with a number only" in low or "reply yes or no only" in low:
        return False
    if _prefer_general_web(low):
        return True
    hints = [
        "latest",
        "recent",
        "current",
        "developments",
        "breakthrough",
        "compare",
        "key features",
        "top ",
        "best ",
        "stock",
        "inflation",
        "market",
        "regulation",
        "launch",
        "specs",
        "tourist attractions",
        "travel tips",
        "ces",
    ]
    if any(h in low for h in hints):
        return True
    # fallback: explicit question with topical nouns
    return ("?" in low) and len(_query_tokens(low)) >= 3


def _ddg_query_variants(query: str) -> List[str]:
    q = normalize(query)
    low = q.lower()
    out = [q]
    intent = _query_intent(q)

    if intent == "specs" and "nvidia" in low and "gpu" in low:
        out += [
            "NVIDIA GeForce RTX 50 series specifications",
            "NVIDIA latest GeForce RTX specs CUDA memory bandwidth",
        ]
    if intent == "compare" and "ada" in low and "blackwell" in low:
        out += [
            "NVIDIA Ada Lovelace vs Blackwell architecture differences",
            "Blackwell compared with Ada Lovelace performance features",
        ]
    if intent == "specs" and "intel" in low and "cpu" in low:
        out += [
            "Intel latest Core Ultra CPUs key features",
            "Intel newest CPU lineup specifications",
        ]
    if intent == "launches" and "spacex" in low:
        out += [
            "latest SpaceX launches and mission outcomes",
            "SpaceX recent missions results",
        ]
    if intent == "travel" and "tokyo" in low:
        out += [
            "Tokyo travel tips 2026",
            "best things to do in Tokyo this year",
        ]
    if intent == "travel" and "london" in low:
        out += [
            "top tourist attractions in London 2026",
            "best places to visit in London this year",
        ]
    if intent == "regulation" and "ai" in low and ("eu" in low or "europe" in low):
        out += [
            "EU AI Act latest developments",
            "European Union AI regulation updates 2026",
        ]
    if intent == "economy" and "inflation" in low:
        out += [
            "current inflation UK US trend",
            "latest inflation data UK and US",
        ]
    if intent == "economy" and ("nvda" in low or "nvidia" in low) and "stock" in low:
        out += [
            "NVDA stock latest performance and news",
            "NVIDIA stock price recent performance",
        ]

    seen = set()
    uniq: List[str] = []
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq


def _latest_query_variants(query: str) -> List[str]:
    q = normalize(query)
    low = q.lower()
    out = [q]
    if "nvidia" in low and "gpu" in low:
        out += [
            "NVIDIA latest GeForce RTX GPUs launch",
            "NVIDIA new GeForce RTX announcement",
        ]
    elif "nvidia" in low and ("latest" in low or "ultime" in low):
        out += [
            "NVIDIA latest GeForce RTX launch",
        ]
    # preserve order but dedupe
    seen = set()
    uniq = []
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq


def _parse_pub_ts(pub_date: str) -> float:
    try:
        return email.utils.parsedate_to_datetime(pub_date).timestamp()
    except Exception:
        return 0.0


def _latest_title_score(title: str, query: str) -> int:
    t = title.lower()
    q = query.lower()
    score = 0
    pos = ["nvidia", "gpu", "geforce", "rtx", "launch", "announc", "series", "blackwell"]
    neg = ["how-to", "guide", "review", "investigating", "fps", "glitches", "opinion"]
    for k in pos:
        if k in t:
            score += 2
    for k in neg:
        if k in t:
            score -= 2
    if "gpu" in q and ("rtx" in t or "geforce" in t):
        score += 3
    if "latest" in q or "ultim" in q:
        if "launch" in t or "announc" in t or "new" in t:
            score += 2
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
            if title and link:
                all_items.append({"title": title, "link": link, "pub_date": pub})

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
    txt = _fetch_text(url, timeout=timeout)
    if not txt:
        return []
    try:
        root = ET.fromstring(txt)
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    for item in root.findall("./channel/item"):
        title = normalize(item.findtext("title", default=""))
        link = normalize(item.findtext("link", default=""))
        pub = normalize(item.findtext("pubDate", default=""))
        if title and link:
            out.append({"title": title, "link": link, "pub_date": pub})
        if len(out) >= max_items:
            break
    return out


def _format_latest_answer(query: str, items: List[Dict[str, str]]) -> str:
    if not items:
        return "I couldn't find recent sources for that query right now."
    titles = [_clean_news_title(it["title"]) for it in items]
    t1 = titles[0].rstrip(".")
    t2 = titles[1].rstrip(".") if len(titles) > 1 else titles[0].rstrip(".")
    resp = f"Recent coverage says: {t1}. Another report says: {t2}."
    src = "\n".join(f"- {it['link']}" for it in items[:3])
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
    return bool(re.search(r"\d+\s*[+\-*/]\s*\d+", u))


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
    ap.add_argument("--web_timeout", type=int, default=15)
    ap.add_argument("--web_results", type=int, default=5)
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
        default=0.18,
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

        if not args.no_web:
            url = _find_first_url(user)
            if url:
                article = _fetch_article_text(url, timeout=args.web_timeout)
                if article:
                    summ = _summarize_plain_text(article, bullets=False)
                    if summ:
                        ans = f"{summ}\n\nSource:\n- {url}"
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

            if _needs_latest_web_search(user) and not _prefer_general_web(user):
                items = _search_google_news(user, max_items=max(1, args.web_results), timeout=args.web_timeout)
                q_score = _latest_news_quality(user, items)
                if q_score < 3:
                    # Fallback to broader web search when RSS news is off-topic/noisy.
                    ddg_items = _search_general_web(
                        user,
                        max_items=max(1, args.web_results),
                        timeout=args.web_timeout,
                    )
                    if ddg_items:
                        ans = _format_general_web_answer(user, ddg_items)
                        print(f"Bot: {ans}\n")
                        _append_turn_log(
                            args.turn_log,
                            user=user,
                            answer=ans,
                            mode="latest_ddg_fallback",
                            source_urls=[it["link"] for it in ddg_items[:3]],
                        )
                        _append_web_log(
                            args.web_log,
                            {
                                "ts_utc": datetime.now(timezone.utc).isoformat(),
                                "mode": "latest_ddg_fallback",
                                "query": user,
                                "source_urls": [it["link"] for it in ddg_items[:3]],
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

            if _needs_general_web_search(user):
                ddg_items = _search_general_web(
                    user,
                    max_items=max(1, args.web_results),
                    timeout=args.web_timeout,
                )
                if ddg_items:
                    ans = _format_general_web_answer(user, ddg_items)
                    print(f"Bot: {ans}\n")
                    _append_turn_log(
                        args.turn_log,
                        user=user,
                        answer=ans,
                        mode="web_general",
                        source_urls=[it["link"] for it in ddg_items[:3]],
                    )
                    _append_web_log(
                        args.web_log,
                        {
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "mode": "web_general",
                            "query": user,
                            "source_urls": [it["link"] for it in ddg_items[:3]],
                            "answer": ans,
                        },
                    )
                    continue

                news_items = _search_google_news(
                    user,
                    max_items=max(1, args.web_results),
                    timeout=args.web_timeout,
                )
                if news_items:
                    ans = _format_latest_answer(user, news_items)
                    print(f"Bot: {ans}\n")
                    _append_turn_log(
                        args.turn_log,
                        user=user,
                        answer=ans,
                        mode="web_general_news_fallback",
                        source_urls=[it["link"] for it in news_items[:3]],
                    )
                    _append_web_log(
                        args.web_log,
                        {
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "mode": "web_general_news_fallback",
                            "query": user,
                            "source_urls": [it["link"] for it in news_items[:3]],
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
        avg_prob = float(meta.get("avg_token_prob", 0.0))

        if (not args.no_web) and (not args.no_confidence_web):
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
                    ddg_items = _search_general_web(
                        user,
                        max_items=max(1, args.web_results),
                        timeout=args.web_timeout,
                    )
                    if ddg_items:
                        web_ans = _format_general_web_answer(user, ddg_items)
                        print(f"Bot: {web_ans}\n")
                        _append_turn_log(
                            args.turn_log,
                            user=user,
                            answer=web_ans,
                            mode="confidence_web_general_fallback",
                            source_urls=[it["link"] for it in ddg_items[:3]],
                            meta={
                                "model_answer": normalize(ans),
                                "model_avg_token_prob": round(avg_prob, 4),
                            },
                        )
                        _append_web_log(
                            args.web_log,
                            {
                                "ts_utc": datetime.now(timezone.utc).isoformat(),
                                "mode": "confidence_web_general_fallback",
                                "query": user,
                                "model_answer": ans,
                                "model_avg_token_prob": round(avg_prob, 4),
                                "source_urls": [it["link"] for it in ddg_items[:3]],
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
            mode="model",
            meta={"model_avg_token_prob": round(avg_prob, 4)},
        )


if __name__ == "__main__":
    main()
