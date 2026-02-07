#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from .config import AssistantConfig
from .engine import GroundedWebAssistant


@dataclass
class EvalCase:
    label: str
    question: str
    url: str
    expect_any: List[str]
    min_sources: int = 1


@dataclass
class EvalResult:
    label: str
    passed: bool
    answer: str
    sources: List[str]


def _default_cases() -> List[EvalCase]:
    return [
        EvalCase(
            label="capital_italy",
            question="What is the capital of Italy?",
            url="https://en.wikipedia.org/wiki/Italy",
            expect_any=["rome"],
            min_sources=1,
        ),
        EvalCase(
            label="berlin_yesno",
            question="Reply YES or NO only: Is Berlin in Germany?",
            url="https://en.wikipedia.org/wiki/Berlin",
            expect_any=["yes"],
            min_sources=1,
        ),
        EvalCase(
            label="language_italy",
            question="What is the official language of Italy?",
            url="https://en.wikipedia.org/wiki/Italy",
            expect_any=["italian"],
            min_sources=1,
        ),
    ]


def _pass(case: EvalCase, answer: str, n_sources: int) -> bool:
    low = answer.lower().strip()
    if n_sources < case.min_sources:
        return False
    return any(x.lower() in low for x in case.expect_any)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grounded web QA evaluator")
    ap.add_argument("--backend", default="hf", choices=["hf", "tiny"])
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
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
    args = ap.parse_args()

    cfg = AssistantConfig(
        backend=args.backend,
        llm_model_name=args.model_name,
        embedding_model_name=args.embedding_model,
        tiny_ckpt=args.tiny_ckpt,
        tiny_tokenizer=args.tiny_tokenizer,
        tiny_lora=args.tiny_lora,
        tiny_top_p=float(args.tiny_top_p),
        temperature=float(args.temperature),
        max_new_tokens=int(args.max_new_tokens),
        top_k=int(args.top_k),
        timeout_sec=int(args.timeout_sec),
    )
    assistant = GroundedWebAssistant(cfg)
    cases = _default_cases()

    results: List[EvalResult] = []
    passed = 0
    for c in cases:
        out = assistant.answer(c.question, url=c.url, search_if_missing=False)
        ok = _pass(c, out.answer, len(out.sources))
        passed += int(ok)
        results.append(EvalResult(label=c.label, passed=ok, answer=out.answer, sources=out.sources))
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
