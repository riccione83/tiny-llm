import re
from typing import List


NOISE_PATTERNS = [
    r"\b(jump to content|main menu|cookie policy|privacy policy|terms of use)\b",
    r"\b(sign in|log in|subscribe|newsletter)\b",
    r"\b(all rights reserved)\b",
]


def normalize_ws(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_web_text(text: str) -> str:
    t = normalize_ws(text)
    lines = [ln.strip() for ln in t.splitlines()]
    out: List[str] = []
    for ln in lines:
        low = ln.lower()
        if len(ln) < 2:
            continue
        if any(re.search(p, low) for p in NOISE_PATTERNS):
            continue
        # Drop lines that look like nav/table artifacts.
        if ln.count("|") >= 3:
            continue
        if re.search(r"^\d+(\.\d+)+\s+[A-Za-z]", ln):
            continue
        out.append(ln)
    return normalize_ws("\n".join(out))


def split_sentences(text: str) -> List[str]:
    t = normalize_ws(text)
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_chars: int = 1100, overlap: int = 180) -> List[str]:
    t = clean_web_text(text)
    if not t:
        return []
    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_chars:
            buf = f"{buf}\n\n{p}".strip() if buf else p
            continue
        if buf:
            chunks.append(buf)
        if len(p) <= chunk_chars:
            buf = p
            continue
        # Long paragraph: hard split by sentence windows.
        sents = split_sentences(p)
        win = ""
        for s in sents:
            if len(win) + len(s) + 1 <= chunk_chars:
                win = f"{win} {s}".strip()
            else:
                if win:
                    chunks.append(win)
                win = s
        if win:
            buf = win
        else:
            buf = ""
    if buf:
        chunks.append(buf)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out: List[str] = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
            continue
        prev_tail = chunks[i - 1][-overlap:].strip()
        out.append(f"{prev_tail}\n\n{c}".strip())
    return out


def truncate_answer(answer: str, max_sentences: int = 3) -> str:
    sents = split_sentences(answer)
    if not sents:
        return normalize_ws(answer)
    return " ".join(sents[:max(1, max_sentences)]).strip()

