#!/usr/bin/env python3
"""
Verify GGUF chat-template metadata and optionally compare to a tokenizer reference.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional


def _load_gguf_reader():
    root = Path(__file__).resolve().parent
    local_gguf_py = root / "tools" / "llama.cpp" / "gguf-py"
    if local_gguf_py.exists():
        sys.path.insert(0, str(local_gguf_py))
    try:
        from gguf import GGUFReader  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Unable to import gguf reader. Ensure llama.cpp repo exists under tools/llama.cpp."
        ) from exc
    return GGUFReader


def _decode_field_bytes(reader, key: str) -> Optional[str]:
    field = reader.fields.get(key)
    if field is None:
        return None
    if not getattr(field, "parts", None):
        return None
    payload = field.parts[-1]
    try:
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        if isinstance(payload, list):
            payload = bytes(payload)
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return str(payload)
    except Exception:
        return None


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_tokenizer_chat_template(model_dir: Path) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    tpl = getattr(tok, "chat_template", None)
    return str(tpl or "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify GGUF tokenizer.chat_template metadata.")
    ap.add_argument("--gguf", required=True, help="Path to GGUF file to verify.")
    ap.add_argument(
        "--reference_model_dir",
        default="",
        help="Optional tokenizer directory to compare chat_template against.",
    )
    ap.add_argument(
        "--write_template",
        default="",
        help="Optional path to write extracted tokenizer.chat_template.",
    )
    args = ap.parse_args()

    gguf_path = Path(args.gguf).resolve()
    if not gguf_path.exists():
        raise SystemExit(f"GGUF not found: {gguf_path}")

    GGUFReader = _load_gguf_reader()
    reader = GGUFReader(str(gguf_path))
    gguf_template = _decode_field_bytes(reader, "tokenizer.chat_template")
    if not gguf_template:
        raise SystemExit("GGUF missing non-empty metadata field: tokenizer.chat_template")

    print(f"[OK] GGUF chat_template found ({len(gguf_template)} chars, sha16={_sha16(gguf_template)})")

    if args.write_template:
        out = Path(args.write_template).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(gguf_template, encoding="utf-8")
        print(f"[OK] Exported chat_template to: {out}")

    if args.reference_model_dir:
        ref_dir = Path(args.reference_model_dir).resolve()
        if not ref_dir.exists():
            raise SystemExit(f"reference_model_dir not found: {ref_dir}")
        ref_template = _load_tokenizer_chat_template(ref_dir)
        if not ref_template:
            raise SystemExit("Reference tokenizer has empty chat_template.")
        if ref_template != gguf_template:
            raise SystemExit(
                "Chat template mismatch: GGUF template differs from reference tokenizer "
                f"(gguf_sha16={_sha16(gguf_template)}, ref_sha16={_sha16(ref_template)})."
            )
        print(
            "[OK] GGUF chat_template matches reference tokenizer "
            f"(sha16={_sha16(ref_template)})"
        )


if __name__ == "__main__":
    main()
