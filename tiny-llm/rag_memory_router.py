#!/usr/bin/env python3
"""
Minimal RAG + memory + local/cloud router.

Local backend: LM Studio OpenAI-compatible endpoint (default http://127.0.0.1:1234/v1/chat/completions)
Cloud backend: OpenAI-compatible endpoint (default https://api.openai.com/v1/chat/completions)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib import error, request


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    text: str
    tokens: Set[str]


def normalize_spaces(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def tokenize_set(text: str) -> Set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text or "")}


def chunk_text(text: str, max_chars: int, overlap_chars: int = 120) -> List[str]:
    clean = normalize_spaces(text)
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    chunks: List[str] = []
    paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip()]
    buf: List[str] = []
    cur_len = 0
    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                chunks.append("\n\n".join(buf))
                buf = []
                cur_len = 0
            start = 0
            while start < len(para):
                end = min(len(para), start + max_chars)
                piece = para[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(para):
                    break
                step = max(1, max_chars - max(0, overlap_chars))
                start += step
            continue
        projected = cur_len + len(para) + (2 if buf else 0)
        if projected > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf = [para]
            cur_len = len(para)
        else:
            buf.append(para)
            cur_len = projected
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _iter_json_like_text_records(path: Path) -> Iterable[str]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    for key in ("text", "content", "output", "answer", "assistant"):
                        val = obj.get(key)
                        if isinstance(val, str) and val.strip():
                            yield val
                    msgs = obj.get("messages")
                    if isinstance(msgs, list):
                        for msg in msgs:
                            if not isinstance(msg, dict):
                                continue
                            role = str(msg.get("role", "")).strip().lower()
                            if role in {"assistant", "system"}:
                                c = msg.get("content")
                                if isinstance(c, str) and c.strip():
                                    yield c
    elif path.suffix.lower() == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(obj, dict):
            obj = [obj]
        if isinstance(obj, list):
            for row in obj:
                if not isinstance(row, dict):
                    continue
                for key in ("text", "content", "output", "answer"):
                    val = row.get(key)
                    if isinstance(val, str) and val.strip():
                        yield val


def _iter_file_chunks(path: Path, max_chars: int) -> Iterable[KnowledgeChunk]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".rst"}:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for piece in chunk_text(text, max_chars=max_chars):
            yield KnowledgeChunk(
                source=str(path),
                text=piece,
                tokens=tokenize_set(piece),
            )
        return
    if suffix in {".jsonl", ".json"}:
        for text_piece in _iter_json_like_text_records(path):
            for piece in chunk_text(text_piece, max_chars=max_chars):
                yield KnowledgeChunk(
                    source=str(path),
                    text=piece,
                    tokens=tokenize_set(piece),
                )


def load_knowledge_chunks(globs: Sequence[str], max_chars: int, max_chunks: int) -> List[KnowledgeChunk]:
    seen: Set[str] = set()
    files: List[Path] = []
    for g in globs:
        for hit in glob.glob(str(g), recursive=True):
            p = Path(hit)
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(p)
    files.sort(key=lambda p: str(p))

    chunks: List[KnowledgeChunk] = []
    for p in files:
        for chunk in _iter_file_chunks(p, max_chars=max_chars):
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def score_chunk(query_tokens: Set[str], chunk: KnowledgeChunk, query: str) -> float:
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(chunk.tokens))
    if overlap == 0:
        return 0.0
    recall = overlap / max(1, len(query_tokens))
    phrase_bonus = 0.0
    lowered = chunk.text.lower()
    for term in query_tokens:
        if len(term) >= 6 and term in lowered:
            phrase_bonus += 0.01
    if query.lower() in lowered:
        phrase_bonus += 0.08
    return recall + min(0.15, phrase_bonus)


def retrieve_context(
    query: str,
    chunks: Sequence[KnowledgeChunk],
    top_k: int,
    min_score: float = 0.05,
) -> List[Tuple[KnowledgeChunk, float]]:
    q_tokens = tokenize_set(query)
    scored: List[Tuple[KnowledgeChunk, float]] = []
    for ch in chunks:
        s = score_chunk(q_tokens, ch, query)
        if s >= min_score:
            scored.append((ch, s))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[: max(0, int(top_k))]


def load_memory_messages(memory_file: Path, max_turns: int) -> List[Dict[str, str]]:
    if not memory_file.exists():
        return []
    rows: List[Dict[str, str]] = []
    with memory_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            user_text = obj.get("user")
            assistant_text = obj.get("assistant")
            if isinstance(user_text, str) and isinstance(assistant_text, str):
                rows.append({"user": user_text, "assistant": assistant_text})
    rows = rows[-max(0, int(max_turns)) :]
    messages: List[Dict[str, str]] = []
    for row in rows:
        messages.append({"role": "user", "content": row["user"]})
        messages.append({"role": "assistant", "content": row["assistant"]})
    return messages


def append_memory_turn(memory_file: Path, user: str, assistant: str, route: str) -> None:
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": int(time.time()),
        "route": str(route),
        "user": str(user),
        "assistant": str(assistant),
    }
    with memory_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def route_query(query: str, mode: str, has_cloud_api_key: bool) -> Tuple[str, str]:
    router_mode = (mode or "auto").strip().lower()
    if router_mode in {"local", "cloud"}:
        if router_mode == "cloud" and not has_cloud_api_key:
            return "local", "cloud API key missing, fallback to local"
        return router_mode, f"forced mode: {router_mode}"

    if not has_cloud_api_key:
        return "local", "auto mode: cloud API key missing"

    q = (query or "").lower()
    local_markers = [
        "python",
        "code",
        "fenced",
        "markdown",
        "json",
        "regex",
        "sql",
        "bash",
        "format",
    ]
    cloud_markers = [
        "tradeoff",
        "architecture",
        "strategy",
        "roadmap",
        "research",
        "compare",
        "multi-step",
        "legal",
        "medical",
        "financial",
    ]
    if any(k in q for k in local_markers):
        return "local", "auto mode: format/code-oriented request"
    if any(k in q for k in cloud_markers):
        return "cloud", "auto mode: deeper reasoning/domain request"
    if len(q) > 420:
        return "cloud", "auto mode: long prompt"
    return "local", "auto mode: default local"


def _coerce_response_text(payload: Dict[str, object]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            msg = c0.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for piece in content:
                        if isinstance(piece, dict):
                            txt = piece.get("text")
                            if isinstance(txt, str):
                                parts.append(txt)
                    if parts:
                        return "\n".join(parts).strip()
            text = c0.get("text")
            if isinstance(text, str):
                return text.strip()
    out_text = payload.get("output_text")
    if isinstance(out_text, str):
        return out_text.strip()
    return ""


def call_chat_api(
    url: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    api_key: str = "",
) -> str:
    body = {
        "messages": list(messages),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if (model or "").strip():
        body["model"] = str(model).strip()

    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    req = request.Request(str(url), data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed request to {url}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc
    text = _coerce_response_text(payload)
    if not text:
        raise RuntimeError(f"No assistant text found in response from {url}")
    return text


def build_messages(
    query: str,
    memory_messages: Sequence[Dict[str, str]],
    retrieved: Sequence[Tuple[KnowledgeChunk, float]],
    system_prompt: str,
) -> List[Dict[str, str]]:
    sys_lines = [normalize_spaces(system_prompt)]
    if retrieved:
        sys_lines.append("Use retrieved context when it is relevant:")
        for idx, (chunk, score) in enumerate(retrieved, start=1):
            snippet = normalize_spaces(chunk.text)
            if len(snippet) > 450:
                snippet = snippet[:450].rstrip() + "..."
            sys_lines.append(f"[{idx}] ({score:.3f}) {chunk.source}: {snippet}")

    messages: List[Dict[str, str]] = [{"role": "system", "content": "\n".join(sys_lines)}]
    messages.extend(memory_messages)
    messages.append({"role": "user", "content": query})
    return messages


def run_single_query(args, query: str, chunks: Sequence[KnowledgeChunk]) -> str:
    memory_file = Path(args.memory_file)
    memory_messages = load_memory_messages(memory_file, max_turns=int(args.memory_turns))
    has_cloud_key = bool(os.getenv(str(args.cloud_api_key_env), "").strip())
    route, reason = route_query(query=query, mode=str(args.router), has_cloud_api_key=has_cloud_key)
    retrieved = retrieve_context(
        query=query,
        chunks=chunks,
        top_k=int(args.knowledge_top_k),
    )
    messages = build_messages(
        query=query,
        memory_messages=memory_messages,
        retrieved=retrieved,
        system_prompt=str(args.system_prompt),
    )

    if route == "cloud":
        api_key = os.getenv(str(args.cloud_api_key_env), "")
        answer = call_chat_api(
            url=str(args.cloud_url),
            model=str(args.cloud_model),
            messages=messages,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout_sec=float(args.timeout_sec),
            api_key=api_key,
        )
    else:
        answer = call_chat_api(
            url=str(args.local_url),
            model=str(args.local_model),
            messages=messages,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout_sec=float(args.timeout_sec),
            api_key="",
        )

    append_memory_turn(memory_file=memory_file, user=query, assistant=answer, route=route)
    if bool(args.show_trace):
        print(f"[route] {route} ({reason})")
        print(f"[retrieved_chunks] {len(retrieved)}")
    return answer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RAG + memory + local/cloud router")
    ap.add_argument("--query", default="", help="Single query mode.")
    ap.add_argument("--interactive", action="store_true", help="Interactive REPL mode.")
    ap.add_argument("--router", default="auto", choices=["auto", "local", "cloud"])
    ap.add_argument("--local_url", default="http://127.0.0.1:1234/v1/chat/completions")
    ap.add_argument("--local_model", default="local-model")
    ap.add_argument("--cloud_url", default="https://api.openai.com/v1/chat/completions")
    ap.add_argument("--cloud_model", default="gpt-4.1-mini")
    ap.add_argument("--cloud_api_key_env", default="OPENAI_API_KEY")
    ap.add_argument(
        "--knowledge_glob",
        action="append",
        default=[],
        help="Knowledge files glob(s). Can be repeated.",
    )
    ap.add_argument("--knowledge_top_k", type=int, default=3)
    ap.add_argument("--knowledge_max_chunks", type=int, default=2000)
    ap.add_argument("--chunk_chars", type=int, default=700)
    ap.add_argument("--memory_file", default="models/chat_memory/session.jsonl")
    ap.add_argument("--memory_turns", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--timeout_sec", type=float, default=120.0)
    ap.add_argument(
        "--system_prompt",
        default=(
            "You are a reliable assistant. Follow user formatting instructions exactly. "
            "If context is insufficient, say so briefly."
        ),
    )
    ap.add_argument("--show_trace", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if (not str(args.query).strip()) and (not bool(args.interactive)):
        raise SystemExit("Provide --query or use --interactive.")

    chunks = load_knowledge_chunks(
        globs=list(args.knowledge_glob),
        max_chars=int(args.chunk_chars),
        max_chunks=int(args.knowledge_max_chunks),
    )
    if bool(args.show_trace):
        print(f"[knowledge_chunks] {len(chunks)}")
        print(f"[memory_file] {Path(args.memory_file).resolve()}")

    if str(args.query).strip():
        answer = run_single_query(args=args, query=str(args.query).strip(), chunks=chunks)
        print(answer)
        return

    print("Interactive mode. Type 'exit' to stop.")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        try:
            answer = run_single_query(args=args, query=query, chunks=chunks)
        except Exception as exc:
            print(f"[error] {exc}")
            continue
        print(answer)


if __name__ == "__main__":
    main()
