#!/usr/bin/env python3
"""
OpenAI-compatible local API server for base and LoRA model variants.

Endpoints:
- GET  /health
- GET  /v1/models
- POST /v1/chat/completions

This server is intentionally minimal and single-process. It lazily loads
models and keeps them cached in memory by model id.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mini_assistant.llm import LocalLLM


ROOT = Path(__file__).resolve().parent


@dataclass
class ModelSpec:
    model_id: str
    mode: str  # base | lora
    model_name_or_path: str
    description: str


def _latest_checkpoint(adapter_root: Path) -> Optional[Path]:
    if not adapter_root.exists() or not adapter_root.is_dir():
        return None
    ckpts = [p for p in adapter_root.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.name.split("-")[-1]))
    return ckpts[-1]


def _recommended_max_3b_adapter() -> Optional[Path]:
    summary = ROOT / "tiny-llm" / "models" / "base_3b_code_max_16gb_v1" / "code_assistant_max_summary.json"
    try:
        if summary.exists():
            obj = json.loads(summary.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                rec = obj.get("recommended")
                if isinstance(rec, dict):
                    adapter = str(rec.get("adapter_dir", "")).strip()
                    if adapter:
                        p = Path(adapter)
                        if p.exists():
                            return p
                        alt = ROOT / "tiny-llm" / p
                        if alt.exists():
                            return alt
    except Exception:
        pass

    # Fallback discovery order when summary is missing.
    for root in [
        ROOT / "tiny-llm" / "models" / "lora3b_code_review_max_repair_v1",
        ROOT / "tiny-llm" / "models" / "lora3b_code_review_max_seed_v1",
    ]:
        latest = _latest_checkpoint(root)
        if latest is not None:
            return latest
    return None


def default_registry() -> Dict[str, ModelSpec]:
    reg: Dict[str, ModelSpec] = {}
    release_7b_merged = ROOT / "tiny-llm" / "models" / "releases" / "tyny-lm-7b-release1" / "merged_model"
    # Base models (HF ids or local dirs).
    reg["base-qwen-0.5b"] = ModelSpec(
        model_id="base-qwen-0.5b",
        mode="base",
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        description="Base Qwen 0.5B Instruct",
    )
    reg["base-qwen-3b"] = ModelSpec(
        model_id="base-qwen-3b",
        mode="base",
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        description="Base Qwen 3B Instruct",
    )
    reg["base-qwen-7b"] = ModelSpec(
        model_id="base-qwen-7b",
        mode="base",
        model_name_or_path=str(ROOT / "tiny-llm" / "models" / "lora_base_7b")
        if (ROOT / "tiny-llm" / "models" / "lora_base_7b").exists()
        else "Qwen/Qwen2.5-7B-Instruct",
        description="Base Qwen 7B Instruct",
    )

    # LoRA variants (auto-discovered when possible).
    # Prefer merged 7B release model for stability in API serving.
    if release_7b_merged.exists():
        reg["tiny-llm-7b"] = ModelSpec(
            model_id="tiny-llm-7b",
            mode="base",
            model_name_or_path=str(release_7b_merged),
            description="tiny-llm 7B merged release model",
        )
    else:
        lora7_latest = _latest_checkpoint(ROOT / "tiny-llm" / "models" / "lora7b_seed_v1")
        if lora7_latest is not None:
            reg["tiny-llm-7b"] = ModelSpec(
                model_id="tiny-llm-7b",
                mode="lora",
                model_name_or_path=str(lora7_latest),
                description="tiny-llm LoRA 7B (latest checkpoint)",
            )

    lora3_latest = _latest_checkpoint(ROOT / "tiny-llm" / "models" / "lora3b_seed_v1")
    if lora3_latest is not None:
        reg["tiny-llm-3b"] = ModelSpec(
            model_id="tiny-llm-3b",
            mode="lora",
            model_name_or_path=str(lora3_latest),
            description="tiny-llm LoRA 3B (latest checkpoint)",
        )

    merged_3b_release = ROOT / "tiny-llm" / "models" / "releases" / "tiny-3b-coding-r1" / "merged_model"
    if merged_3b_release.exists() and (merged_3b_release / "config.json").exists():
        reg["tiny-3b-coding"] = ModelSpec(
            model_id="tiny-3b-coding",
            mode="base",
            model_name_or_path=str(merged_3b_release),
            description="tiny 3B coding assistant merged release r1",
        )
    else:
        coding_adapter = _recommended_max_3b_adapter()
        if coding_adapter is not None:
            reg["tiny-3b-coding"] = ModelSpec(
                model_id="tiny-3b-coding",
                mode="lora",
                model_name_or_path=str(coding_adapter),
                description="tiny 3B coding assistant (recommended checkpoint)",
            )
        else:
            cpt_base = ROOT / "tiny-llm" / "models" / "base_3b_code_max_16gb_v1"
            if cpt_base.exists() and (cpt_base / "config.json").exists():
                reg["tiny-3b-coding"] = ModelSpec(
                    model_id="tiny-3b-coding",
                    mode="base",
                    model_name_or_path=str(cpt_base),
                    description="tiny 3B coding assistant base (CPT)",
                )
            else:
                reg["tiny-3b-coding"] = ModelSpec(
                    model_id="tiny-3b-coding",
                    mode="base",
                    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
                    description="tiny 3B coding assistant (fallback to Qwen 3B base)",
                )

    lora05_latest = _latest_checkpoint(ROOT / "tiny-llm" / "models" / "lora_repair_v2")
    if lora05_latest is not None:
        reg["tiny-llm-0.5b"] = ModelSpec(
            model_id="tiny-llm-0.5b",
            mode="lora",
            model_name_or_path=str(lora05_latest),
            description="tiny-llm LoRA 0.5B (latest checkpoint)",
        )
    return reg


def load_registry(registry_path: str) -> Dict[str, ModelSpec]:
    default = default_registry()
    if not registry_path:
        return default
    p = Path(registry_path)
    if not p.exists():
        return default
    obj = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, ModelSpec] = dict(default)
    for row in obj.get("models", []):
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id", "")).strip()
        mode = str(row.get("mode", "base")).strip().lower()
        model_path = str(row.get("model_name_or_path", "")).strip()
        if not model_id or not model_path or mode not in {"base", "lora"}:
            continue
        out[model_id] = ModelSpec(
            model_id=model_id,
            mode=mode,
            model_name_or_path=model_path,
            description=str(row.get("description", "")).strip() or model_id,
        )
    return out


def _normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    out: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object")
        role = str(m.get("role", "")).strip().lower()
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        if not isinstance(content, str):
            raise ValueError("message content must be string")
        out.append({"role": role, "content": content})
    return out


def _count_tokens(llm: LocalLLM, text: str) -> int:
    try:
        impl = getattr(llm, "_impl", None)
        tok = getattr(impl, "_tok", None)
        if tok is not None:
            ids = tok(text, return_tensors=None).get("input_ids", [])
            if isinstance(ids, list):
                return len(ids)
    except Exception:
        pass
    return max(1, len(text.split()))


def _message_token_len(llm: LocalLLM, messages: List[Dict[str, str]], use_chat_template: bool) -> int:
    try:
        impl = getattr(llm, "_impl", None)
        tok = getattr(impl, "_tok", None)
        if tok is None:
            raise RuntimeError("tokenizer unavailable")
        if use_chat_template and hasattr(tok, "apply_chat_template"):
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if isinstance(ids, list):
                return int(len(ids))
        prompt = "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        out = tok(prompt, return_tensors=None).get("input_ids", [])
        if isinstance(out, list):
            return int(len(out))
    except Exception:
        pass
    fallback = "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
    return max(1, len(fallback.split()))


def _truncate_messages_to_token_budget(
    llm: LocalLLM,
    messages: List[Dict[str, str]],
    max_prompt_tokens: int,
    use_chat_template: bool,
) -> Tuple[List[Dict[str, str]], int, int]:
    if max_prompt_tokens <= 0:
        cur = _message_token_len(llm, messages, use_chat_template=use_chat_template)
        return messages, 0, cur
    kept = list(messages)
    before = _message_token_len(llm, kept, use_chat_template=use_chat_template)
    if before <= max_prompt_tokens:
        return kept, 0, before
    removed = 0

    # Always preserve the latest non-system user/assistant message if present.
    latest_non_system_idx = None
    for i in range(len(kept) - 1, -1, -1):
        if str(kept[i].get("role", "")).lower() != "system":
            latest_non_system_idx = i
            break

    # Phase 1: drop oldest non-system messages, but never the latest non-system.
    while True:
        cur = _message_token_len(llm, kept, use_chat_template=use_chat_template)
        if cur <= max_prompt_tokens:
            return kept, removed, cur
        drop_idx = next(
            (
                i
                for i, m in enumerate(kept)
                if str(m.get("role", "")).lower() != "system" and i != latest_non_system_idx
            ),
            None,
        )
        if drop_idx is None:
            break
        kept.pop(drop_idx)
        removed += 1
        if latest_non_system_idx is not None and drop_idx < latest_non_system_idx:
            latest_non_system_idx -= 1

    # Phase 2: if still too large, drop oldest system messages.
    while True:
        cur = _message_token_len(llm, kept, use_chat_template=use_chat_template)
        if cur <= max_prompt_tokens:
            return kept, removed, cur
        drop_idx = next((i for i, m in enumerate(kept) if str(m.get("role", "")).lower() == "system"), None)
        if drop_idx is None:
            break
        kept.pop(drop_idx)
        removed += 1
        if latest_non_system_idx is not None and drop_idx < latest_non_system_idx:
            latest_non_system_idx -= 1

    # Phase 3: if only one oversized message remains, hard-truncate its content.
    cur = _message_token_len(llm, kept, use_chat_template=use_chat_template)
    if cur > max_prompt_tokens and len(kept) == 1:
        msg = kept[0]
        content = str(msg.get("content", ""))
        if content:
            # Coarse truncation by chars to enforce budget quickly.
            ratio = max(0.05, float(max_prompt_tokens) / float(max(1, cur)))
            new_len = max(64, int(len(content) * ratio))
            msg["content"] = content[-new_len:]
        cur = _message_token_len(llm, kept, use_chat_template=use_chat_template)
    return kept, removed, cur


class ServerState:
    def __init__(
        self,
        registry: Dict[str, ModelSpec],
        default_model: str,
        use_chat_template: bool,
        http_debug: bool = False,
        http_debug_max_chars: int = 2000,
        max_prompt_tokens: int = 1024,
        fixed_response_text: str = "",
        max_completion_tokens_cap: int = 512,
        max_request_bytes: int = 32768,
    ) -> None:
        self.registry = registry
        self.default_model = default_model
        self.use_chat_template = bool(use_chat_template)
        self.http_debug = bool(http_debug)
        self.http_debug_max_chars = max(200, int(http_debug_max_chars))
        self.max_prompt_tokens = max(0, int(max_prompt_tokens))
        self.fixed_response_text = str(fixed_response_text or "").strip()
        self.max_completion_tokens_cap = max(1, int(max_completion_tokens_cap))
        self.max_request_bytes = max(1024, int(max_request_bytes))
        self.cache: Dict[str, LocalLLM] = {}

    def resolve_model(self, model_id: str) -> Tuple[str, ModelSpec]:
        mid = (model_id or "").strip() or self.default_model
        if mid not in self.registry:
            raise KeyError(f"Unknown model '{mid}'. Use GET /v1/models.")
        return mid, self.registry[mid]

    def get_llm(self, model_id: str, spec: ModelSpec, temperature: float, max_new_tokens: int) -> LocalLLM:
        # Keep one loaded instance per model id to avoid repeated large-model
        # dispatch/offload churn across requests.
        key = f"{model_id}|{int(self.use_chat_template)}"
        if key in self.cache:
            llm = self.cache[key]
            self._apply_runtime_generation_settings(llm, temperature=temperature, max_new_tokens=max_new_tokens)
            return llm
        llm = LocalLLM(
            backend="hf",
            model_name=spec.model_name_or_path,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            use_chat_template=self.use_chat_template,
        )
        self._apply_runtime_generation_settings(llm, temperature=temperature, max_new_tokens=max_new_tokens)
        self.cache[key] = llm
        return llm

    @staticmethod
    def _apply_runtime_generation_settings(llm: LocalLLM, temperature: float, max_new_tokens: int) -> None:
        # LocalLLM wraps either HFLocalLLM or TinyLocalLLM. For this API server
        # we only use HF backend, but keep this defensive.
        impl = getattr(llm, "_impl", None)
        if impl is None:
            return
        if hasattr(impl, "temperature"):
            try:
                impl.temperature = float(temperature)
            except Exception:
                pass
        if hasattr(impl, "max_new_tokens"):
            try:
                impl.max_new_tokens = int(max_new_tokens)
            except Exception:
                pass


def make_handler(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _dbg(self, msg: str) -> None:
            if state.http_debug:
                print(f"[http] {msg}")

        def _truncate(self, s: str) -> str:
            lim = int(state.http_debug_max_chars)
            if len(s) <= lim:
                return s
            return s[:lim] + f"... <truncated {len(s) - lim} chars>"

        def _send_json(self, status: int, payload: Dict[str, Any]) -> bool:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return True
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # Client closed socket before response was written.
                return False

        def _send_sse_headers(self, status: int = 200) -> bool:
            try:
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                return True
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return False

        def _sse_write_json(self, obj: Dict[str, Any]) -> bool:
            try:
                data = json.dumps(obj, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return False

        def _sse_write_done(self) -> bool:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return False

        def do_GET(self) -> None:
            self._dbg(f"{self.command} {self.path}")
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            if self.path == "/v1/models":
                rows = [
                    {"id": spec.model_id, "object": "model", "owned_by": "local", "mode": spec.mode}
                    for spec in state.registry.values()
                ]
                self._send_json(200, {"object": "list", "data": rows})
                return
            self._send_json(404, {"error": {"message": "Not found", "type": "not_found"}})

        def do_POST(self) -> None:
            self._dbg(f"{self.command} {self.path}")
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "Not found", "type": "not_found"}})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > int(state.max_request_bytes):
                    self._send_json(
                        400,
                        {
                            "error": {
                                "message": (
                                    f"Request body too large: {n} bytes. "
                                    f"Server limit is {state.max_request_bytes} bytes."
                                ),
                                "type": "invalid_request_error",
                            }
                        },
                    )
                    return
                raw = self.rfile.read(n)
                self._dbg(f"request_bytes={n}")
                self._dbg(f"request_body={self._truncate(raw.decode('utf-8', errors='replace'))}")
                body = json.loads(raw.decode("utf-8"))
                model = str(body.get("model", "")).strip()
                messages = _normalize_messages(body.get("messages", []))
                temperature = float(body.get("temperature", 0.0))
                max_tokens = int(body.get("max_tokens", 300))
                if max_tokens > int(state.max_completion_tokens_cap):
                    self._dbg(
                        f"max_tokens clamped from {max_tokens} to {state.max_completion_tokens_cap}"
                    )
                    max_tokens = int(state.max_completion_tokens_cap)
                stream = bool(body.get("stream", False))
                stream_options = body.get("stream_options", {}) if isinstance(body.get("stream_options"), dict) else {}
                include_usage = bool(stream_options.get("include_usage", False))
                model_id, spec = state.resolve_model(model)
                llm = None
                if state.fixed_response_text:
                    # Debug mode: bypass model to validate client protocol handling.
                    out = state.fixed_response_text
                    latency_ms = 0
                    prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    prompt_tokens = max(1, len(prompt_text.split()))
                    completion_tokens = max(1, len(out.split()))
                    print(f"[fixed] model={model_id} completion_tokens={completion_tokens}")
                else:
                    llm = state.get_llm(
                        model_id=model_id,
                        spec=spec,
                        temperature=temperature,
                        max_new_tokens=max_tokens,
                    )
                    original_msg_count = len(messages)
                    messages, removed_count, prompt_tokens_after_trim = _truncate_messages_to_token_budget(
                        llm=llm,
                        messages=messages,
                        max_prompt_tokens=int(state.max_prompt_tokens),
                        use_chat_template=bool(state.use_chat_template),
                    )
                    if removed_count > 0:
                        print(
                            f"[trim] model={model_id} removed_messages={removed_count} "
                            f"orig_messages={original_msg_count} prompt_tokens_after_trim={prompt_tokens_after_trim} "
                            f"max_prompt_tokens={state.max_prompt_tokens}"
                        )
                    prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    t0 = time.perf_counter()
                    out = llm.generate(messages)
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    impl = getattr(llm, "_impl", None)
                    prof = getattr(impl, "_last_profile", {}) if impl is not None else {}
                    if isinstance(prof, dict) and prof:
                        print(
                            "[perf] "
                            f"model={model_id} "
                            f"tokenize_ms={prof.get('tokenize_ms')} "
                            f"generate_ms={prof.get('generate_ms')} "
                            f"prompt_tokens={prof.get('prompt_tokens')} "
                            f"tokens_generated={prof.get('tokens_generated')} "
                            f"tok_s={prof.get('tokens_per_sec')} "
                            f"device={prof.get('device')} "
                            f"dtype={prof.get('dtype')} "
                            f"gpu_mem_mb={prof.get('gpu_mem_allocated_mb')}"
                        )
                    prompt_tokens = _count_tokens(llm, prompt_text)
                    completion_tokens = _count_tokens(llm, out)
                if stream:
                    created = int(time.time())
                    stream_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                    ok = self._send_sse_headers(200)
                    if not ok:
                        print(f"[client] disconnected before stream headers were sent: model={model_id}")
                        return
                    first_chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                    if not self._sse_write_json(first_chunk):
                        print(f"[client] disconnected during stream (role chunk): model={model_id}")
                        return
                    impl = getattr(llm, "_impl", None) if llm is not None else None
                    token_like_pieces = getattr(impl, "_last_stream_pieces", None) if impl is not None else None
                    if isinstance(token_like_pieces, list) and token_like_pieces:
                        pieces_iter = token_like_pieces
                    else:
                        chunk_size = 64
                        pieces_iter = [out[i : i + chunk_size] for i in range(0, len(out), chunk_size)]
                    for piece in pieces_iter:
                        if not piece:
                            continue
                        chunk = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                        }
                        if not self._sse_write_json(chunk):
                            print(f"[client] disconnected during stream (content chunk): model={model_id}")
                            return
                    last_chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    if not self._sse_write_json(last_chunk):
                        print(f"[client] disconnected during stream (final chunk): model={model_id}")
                        return
                    if include_usage:
                        usage_chunk = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        }
                        if not self._sse_write_json(usage_chunk):
                            print(f"[client] disconnected during stream (usage chunk): model={model_id}")
                            return
                    self._sse_write_done()
                    self.close_connection = True
                    self._dbg(
                        f"response_status=200 stream=true model={model_id} latency_ms={latency_ms} "
                        f"resp_chars={len(out)}"
                    )
                    return
                payload = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": out},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "model_used": {"id": model_id, "mode": spec.mode, "source": spec.model_name_or_path},
                    "latency_ms": latency_ms,
                    "tokens": prompt_tokens + completion_tokens,
                }
                sent = self._send_json(200, payload)
                self._dbg(
                    f"response_status=200 model={model_id} latency_ms={latency_ms} "
                    f"resp_chars={len(out)}"
                )
                if state.http_debug:
                    self._dbg(f"response_body={self._truncate(json.dumps(payload, ensure_ascii=False))}")
                if not sent:
                    print(f"[client] disconnected before response was sent: model={model_id}")
            except KeyError as exc:
                self._dbg(f"response_status=400 error={exc}")
                self._send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except ValueError as exc:
                self._dbg(f"response_status=400 error={exc}")
                self._send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception as exc:
                self._dbg(f"response_status=500 error={type(exc).__name__}: {exc}")
                self._send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}})

        def log_message(self, fmt: str, *args: Any) -> None:
            # keep stdout clean by default
            return

    return Handler


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OpenAI-compatible local server for base/LoRA variants")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--registry_json", default="", help="Optional model registry JSON override")
    ap.add_argument("--default_model", default="tiny-llm-7b")
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--no_preload_default", action="store_true", help="Disable default startup preload")
    ap.add_argument("--warmup_tokens", type=int, default=16, help="Warmup generation tokens at startup (default: 16)")
    ap.add_argument("--no_warmup", action="store_true", help="Disable startup warmup generation")
    ap.add_argument("--http_debug", action="store_true", help="Print HTTP request/response debug logs")
    ap.add_argument("--http_debug_max_chars", type=int, default=2000, help="Max chars to print for request/response bodies")
    ap.add_argument("--max_prompt_tokens", type=int, default=1024, help="Max input prompt tokens (0 disables trimming)")
    ap.add_argument("--fixed_response_text", default="", help="If set, bypass model and always return this text")
    ap.add_argument(
        "--max_completion_tokens_cap",
        type=int,
        default=512,
        help="Server-side hard cap for max_tokens requested by clients",
    )
    ap.add_argument(
        "--max_request_bytes",
        type=int,
        default=32768,
        help="Reject HTTP request bodies larger than this size (bytes)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(str(args.registry_json))
    if not registry:
        raise SystemExit("No models available in registry.")
    if str(args.default_model) not in registry:
        # fallback to first model in registry order
        args.default_model = next(iter(registry.keys()))
    state = ServerState(
        registry=registry,
        default_model=str(args.default_model),
        use_chat_template=(not bool(args.no_chat_template)),
        http_debug=bool(args.http_debug),
        http_debug_max_chars=int(args.http_debug_max_chars),
        max_prompt_tokens=int(args.max_prompt_tokens),
        fixed_response_text=str(args.fixed_response_text),
        max_completion_tokens_cap=int(args.max_completion_tokens_cap),
        max_request_bytes=int(args.max_request_bytes),
    )
    do_preload = (not bool(args.no_preload_default)) and (not bool(state.fixed_response_text))
    warmup_tokens = 0 if bool(args.no_warmup) else max(0, int(args.warmup_tokens))
    if do_preload:
        try:
            model_id, spec = state.resolve_model(str(args.default_model))
            t0 = time.perf_counter()
            llm = state.get_llm(
                model_id=model_id,
                spec=spec,
                temperature=0.0,
                max_new_tokens=max(16, warmup_tokens if warmup_tokens > 0 else 16),
            )
            load_ms = int((time.perf_counter() - t0) * 1000)
            print(f"[startup] preloaded model={model_id} in {load_ms} ms")
            if warmup_tokens > 0:
                w0 = time.perf_counter()
                _ = llm.generate([{"role": "user", "content": "Reply with OK."}])
                warm_ms = int((time.perf_counter() - w0) * 1000)
                print(f"[startup] warmup generate completed in {warm_ms} ms")
        except Exception as exc:
            print(f"[startup] preload failed: {type(exc).__name__}: {exc}")
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), handler)
    print(f"Model API server listening on http://{args.host}:{args.port}")
    print("Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions")
    print(f"Default model: {args.default_model}")
    print("[ready] server is ready to accept requests")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
