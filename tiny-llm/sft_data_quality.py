#!/usr/bin/env python3
"""
Strict JSONL validation and minimal SFT hygiene helpers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_JSON_DECODER = json.JSONDecoder()
_NO_MARKDOWN_RE = re.compile(
    r"\b("
    r"no\s+markdown|without\s+markdown|plain\s*text\s*only|"
    r"output\s+only\s+text|do\s+not\s+use\s+markdown|don't\s+use\s+markdown|"
    r"no\s+code\s+fences?|without\s+fences?"
    r")\b",
    flags=re.IGNORECASE,
)
_PYTHON_PATTERNS = [
    re.compile(r"(^|\n)\s*def\s+[A-Za-z_]\w*\s*\(", flags=re.MULTILINE),
    re.compile(r"(^|\n)\s*class\s+[A-Za-z_]\w*\s*[:(]", flags=re.MULTILINE),
    re.compile(r"(^|\n)\s*from\s+[A-Za-z_][\w.]*\s+import\s+", flags=re.MULTILINE),
    re.compile(r"(^|\n)\s*import\s+[A-Za-z_][\w.]*", flags=re.MULTILINE),
    re.compile(r"(^|\n)\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", flags=re.MULTILINE),
    re.compile(r"(^|\n)\s*(?:for|while|if|elif|try|except|with)\b.*:", flags=re.MULTILINE),
    re.compile(r"(^|\n)\s*print\s*\(", flags=re.MULTILINE),
    re.compile(r"(^|\n)\s*[A-Za-z_]\w*\s*=\s*.+", flags=re.MULTILINE),
]
_PYTHON_FENCE_RE = re.compile(r"```[ \t]*python\b", flags=re.IGNORECASE)


@dataclass
class JsonlLineError:
    line_no: int
    message: str


@dataclass
class JsonlFileReport:
    path: Path
    total_lines: int = 0
    valid_lines: int = 0
    invalid_lines: int = 0
    loaded_examples: int = 0
    filtered_examples: int = 0
    duplicate_examples: int = 0
    errors: List[JsonlLineError] = field(default_factory=list)


@dataclass
class SftHygieneConfig:
    code_fence_mode: str = "off"  # off | reject | normalize
    fence_language: str = "python"
    reject_no_markdown_code_instructions: bool = True


@dataclass
class SftHygieneResult:
    keep: bool
    answer: str
    reason: str = ""


def normalize_text_preserve_whitespace(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_jsonl_object_line(raw_line: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    line = (raw_line or "").strip()
    if not line:
        return None, "blank line (expected exactly one JSON object)"

    if re.search(r"}\s*{", line):
        return None, "multiple JSON objects on one line (detected `}{`/`} {` pattern)"

    try:
        obj, end = _JSON_DECODER.raw_decode(line)
    except json.JSONDecodeError as exc:
        msg = f"{exc.msg} (column {exc.colno})"
        if "Unterminated string" in exc.msg:
            msg += "; possible unescaped literal newline in JSON string (use \\n)"
        return None, msg

    tail = line[end:].strip()
    if tail:
        if tail.startswith("{"):
            return None, "multiple JSON objects on one line"
        return None, f"extra content after JSON object: {tail[:80]!r}"

    if not isinstance(obj, dict):
        return None, f"top-level JSON value must be an object, got {type(obj).__name__}"
    return obj, None


def row_has_no_markdown_instruction(row: Dict[str, object]) -> bool:
    chunks: List[str] = []
    for key in ("system_prompt", "instruction", "input", "context", "question", "prompt"):
        value = row.get(key)
        if isinstance(value, str):
            chunks.append(value)

    messages = row.get("messages")
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).strip().lower()
            if role in {"system", "user"}:
                chunks.append(str(m.get("content", "")))
    text = "\n".join(chunks)
    return bool(_NO_MARKDOWN_RE.search(text))


def looks_like_python_code(text: str) -> bool:
    t = normalize_text_preserve_whitespace(text)
    if not t:
        return False
    if "```" in t and _PYTHON_FENCE_RE.search(t):
        return True
    if "\n" not in t and len(t) < 24:
        return False
    return any(p.search(t) for p in _PYTHON_PATTERNS)


def _has_balanced_fences(text: str) -> bool:
    cnt = text.count("```")
    return cnt >= 2 and (cnt % 2 == 0)


def _strip_fence_markers(text: str) -> str:
    t = normalize_text_preserve_whitespace(text)
    t = re.sub(r"```[A-Za-z0-9_-]*", "", t)
    return t.strip()


def _wrap_python_fence(text: str, language: str) -> str:
    body = _strip_fence_markers(text)
    return f"```{language}\n{body}\n```"


def apply_sft_hygiene(
    row: Dict[str, object],
    answer: str,
    cfg: SftHygieneConfig,
) -> SftHygieneResult:
    out = normalize_text_preserve_whitespace(answer)
    if not out:
        return SftHygieneResult(keep=False, answer="", reason="empty_answer")

    mode = (cfg.code_fence_mode or "off").strip().lower()
    if mode not in {"off", "reject", "normalize"}:
        raise ValueError(f"Unsupported code_fence_mode: {cfg.code_fence_mode}")
    if mode == "off":
        return SftHygieneResult(keep=True, answer=out)

    code_like = looks_like_python_code(out)
    if not code_like:
        return SftHygieneResult(keep=True, answer=out)

    if cfg.reject_no_markdown_code_instructions and row_has_no_markdown_instruction(row):
        return SftHygieneResult(keep=False, answer=out, reason="conflicting_no_markdown_instruction")

    has_fence = "```" in out
    has_python_fence = bool(_PYTHON_FENCE_RE.search(out))
    balanced = _has_balanced_fences(out) if has_fence else False
    if has_fence and has_python_fence and balanced:
        return SftHygieneResult(keep=True, answer=out)

    if mode == "reject":
        if has_fence and not balanced:
            reason = "broken_code_fence"
        elif has_fence and not has_python_fence:
            reason = "missing_python_fence_language"
        else:
            reason = "missing_code_fence"
        return SftHygieneResult(keep=False, answer=out, reason=reason)

    normalized = _wrap_python_fence(out, language=(cfg.fence_language or "python").strip().lower() or "python")
    return SftHygieneResult(keep=True, answer=normalized, reason="normalized_python_fence")


def format_validation_table(reports: Sequence[JsonlFileReport]) -> str:
    headers = (
        "file",
        "total_lines",
        "valid_lines",
        "invalid_lines",
        "loaded_examples",
        "filtered_examples",
        "duplicate_examples",
    )
    rows = [headers]
    for r in reports:
        rows.append(
            (
                str(r.path),
                str(r.total_lines),
                str(r.valid_lines),
                str(r.invalid_lines),
                str(r.loaded_examples),
                str(r.filtered_examples),
                str(r.duplicate_examples),
            )
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    rendered = []
    for idx, row in enumerate(rows):
        rendered.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if idx == 0:
            rendered.append("-+-".join("-" * w for w in widths))
    return "\n".join(rendered)
