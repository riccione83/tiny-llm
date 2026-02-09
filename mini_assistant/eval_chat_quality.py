#!/usr/bin/env python3
"""
Lightweight chat-quality regression eval (offline).

Goal:
- Fast sanity checks for instruction following and formatting.
- Works with both backends:
  - hf: any HF model id or local HF folder
  - tiny: legacy custom checkpoint (optional)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .llm import LocalLLM


def _norm(s: str) -> str:
    return (s or "").strip()


def _lower(s: str) -> str:
    return _norm(s).lower()


@dataclass
class EvalCase:
    label: str
    messages: List[Dict[str, str]]
    expect_regex: Optional[str] = None
    expect_any: Optional[List[str]] = None
    expect_json: bool = False
    expect_json_fields: Optional[Dict[str, str]] = None


@dataclass
class EvalResult:
    label: str
    passed: bool
    answer: str
    debug: Dict[str, object]


def _default_cases() -> List[EvalCase]:
    return [
        EvalCase(
            label="greeting_short",
            messages=[
                {"role": "user", "content": "Say hello in one short sentence."},
            ],
            expect_regex=r"\\b(hello|hi|ciao|salve)\\b",
        ),
        EvalCase(
            label="yesno_strict",
            messages=[
                {"role": "user", "content": "Reply YES or NO only: Is Berlin in Germany?"},
            ],
            expect_regex=r"^(yes|no)\\.?$",
        ),
        EvalCase(
            label="math_only_number",
            messages=[
                {"role": "user", "content": "Quanto fa 144/12? Rispondi solo con un numero."},
            ],
            expect_regex=r"^12$",
        ),
        EvalCase(
            label="json_only",
            messages=[
                {
                    "role": "user",
                    "content": "Output ONLY valid JSON (no markdown). Keys: name, language. Values: Italy, Italian.",
                },
            ],
            expect_json=True,
            expect_json_fields={"name": "Italy", "language": "Italian"},
        ),
        EvalCase(
            label="binary_search_complexity",
            messages=[
                {
                    "role": "user",
                    "content": "Explain binary search in simple words and state its time complexity.",
                },
            ],
            expect_regex=r"(o\\(log\\s*n\\)|logarithmic|log\\s*n)",
        ),
        EvalCase(
            label="python_palindrome",
            messages=[
                {
                    "role": "user",
                    "content": "Write a short Python function is_palindrome(s: str) -> bool. Keep it minimal.",
                },
            ],
            expect_regex=r"def\\s+is_palindrome\\b",
        ),
    ]


def _extract_json_str(text: str) -> Optional[str]:
    t = _norm(text)
    if not t:
        return None
    if t.startswith("{") and t.endswith("}"):
        return t
    # Be helpful but still strict-ish: try to locate a JSON object in the output.
    start = t.find("{")
    end = t.rfind("}")
    if 0 <= start < end:
        candidate = t[start : end + 1].strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    return None


def _check_case(case: EvalCase, answer: str) -> (bool, Dict[str, object]):
    dbg: Dict[str, object] = {}
    txt = _norm(answer)
    if not txt:
        return False, {"reason": "empty"}

    if case.expect_json:
        js = _extract_json_str(txt)
        if js is None:
            return False, {"reason": "no_json_object_found"}
        try:
            obj = json.loads(js)
        except Exception as exc:
            return False, {"reason": "json_parse_error", "error": str(exc)}
        if not isinstance(obj, dict):
            return False, {"reason": "json_not_object"}
        dbg["json"] = obj
        if case.expect_json_fields:
            for k, v in case.expect_json_fields.items():
                if str(obj.get(k, "")).strip() != str(v):
                    return False, {"reason": "json_field_mismatch", "field": k, "got": obj.get(k), "want": v}

    if case.expect_any:
        low = _lower(txt)
        if not any(str(x).lower() in low for x in case.expect_any):
            return False, {"reason": "missing_expected_substring", "expect_any": case.expect_any}

    if case.expect_regex:
        pat = re.compile(case.expect_regex, flags=re.IGNORECASE | re.MULTILINE)
        if not pat.search(txt):
            return False, {"reason": "regex_mismatch", "pattern": case.expect_regex}

    return True, dbg


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline chat-quality regression evaluator")
    ap.add_argument("--backend", default="hf", choices=["hf", "tiny"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--tiny_ckpt", default="checkpoints_v2/final.pt")
    ap.add_argument("--tiny_tokenizer", default="tokenizer.model")
    ap.add_argument("--tiny_lora", default="")
    ap.add_argument("--tiny_top_p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--out_json", default="data/eval_chat_quality_report.json")
    args = ap.parse_args()

    llm = LocalLLM(
        backend=args.backend,
        model_name=args.model_name,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        tiny_ckpt=args.tiny_ckpt,
        tiny_tokenizer=args.tiny_tokenizer,
        tiny_lora=args.tiny_lora,
        tiny_top_p=float(args.tiny_top_p),
    )
    cases = _default_cases()

    results: List[EvalResult] = []
    passed = 0
    for c in cases:
        ans = llm.generate(c.messages)
        ok, dbg = _check_case(c, ans)
        passed += int(ok)
        results.append(EvalResult(label=c.label, passed=ok, answer=ans, debug=dbg))
        print(f"[{'OK' if ok else 'FAIL'}] {c.label}")

    total = len(cases)
    print(f"PASS {passed}/{total}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": passed,
        "total": total,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {out_path}")

    if passed < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

