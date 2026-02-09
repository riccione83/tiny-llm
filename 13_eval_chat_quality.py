#!/usr/bin/env python3
"""
Compatibility wrapper for project eval.

Old command style is still accepted, but evaluation now runs the new
pipelines:
- grounded: mini_assistant/eval_grounded.py (web QA regression)
- chat:     mini_assistant/eval_chat_quality.py (offline chat sanity)
"""

import argparse
import subprocess
import sys
from typing import List


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="grounded", choices=["grounded", "chat", "both"])
    # New args
    ap.add_argument("--backend", default="hf", choices=["hf", "tiny"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tiny_ckpt", default="checkpoints_v2/final.pt")
    ap.add_argument("--tiny_tokenizer", default="tokenizer.model")
    ap.add_argument("--tiny_lora", default="")
    ap.add_argument("--tiny_top_p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--timeout_sec", type=int, default=20)
    ap.add_argument("--out_json", default="data/eval_grounded_report.json")

    # Legacy args kept for compatibility; ignored in new evaluator.
    ap.add_argument("--base_ckpt", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--lora_adapter", default="")
    ap.add_argument("--chat_script", default="")
    ap.add_argument("--web_results", type=int, default=3)
    ap.add_argument("--python", default="")

    args = ap.parse_args()

    rc = 0
    if args.suite in {"grounded", "both"}:
        cmd: List[str] = [
            sys.executable,
            "-m",
            "mini_assistant.eval_grounded",
            "--backend",
            args.backend,
            "--model_name",
            args.model_name,
            "--embedding_model",
            args.embedding_model,
            "--tiny_ckpt",
            args.tiny_ckpt,
            "--tiny_tokenizer",
            args.tiny_tokenizer,
            "--tiny_lora",
            args.tiny_lora,
            "--tiny_top_p",
            str(args.tiny_top_p),
            "--temperature",
            str(args.temperature),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--top_k",
            str(args.top_k),
            "--timeout_sec",
            str(args.timeout_sec),
            "--out_json",
            args.out_json,
        ]
        p = subprocess.run(cmd)
        rc = max(rc, int(p.returncode))

    if args.suite in {"chat", "both"}:
        cmd2: List[str] = [
            sys.executable,
            "-m",
            "mini_assistant.eval_chat_quality",
            "--backend",
            args.backend,
            "--model_name",
            args.model_name,
            "--tiny_ckpt",
            args.tiny_ckpt,
            "--tiny_tokenizer",
            args.tiny_tokenizer,
            "--tiny_lora",
            args.tiny_lora,
            "--tiny_top_p",
            str(args.tiny_top_p),
            "--temperature",
            str(args.temperature),
            "--max_new_tokens",
            str(args.max_new_tokens),
        ]
        p2 = subprocess.run(cmd2)
        rc = max(rc, int(p2.returncode))

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
