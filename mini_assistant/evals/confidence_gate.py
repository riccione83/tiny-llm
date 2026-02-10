#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from ..config import AssistantConfig
from ..engine import GroundedWebAssistant


@dataclass
class GateCase:
    label: str
    question: str
    expected_route: str
    url: str = ""
    answer_must_contain: Optional[str] = None


@dataclass
class GateResult:
    label: str
    passed: bool
    expected_route: str
    route: str
    answer: str
    sources: List[str]
    debug: dict


def default_cases() -> List[GateCase]:
    return [
        GateCase("smalltalk_hi", "Hi", expected_route="direct"),
        GateCase("direct_math", "What is 2+2?", expected_route="direct", answer_must_contain="4"),
        GateCase("direct_capital", "What is the capital of Italy?", expected_route="direct", answer_must_contain="rome"),
        GateCase("web_latest", "What is the latest nvidia gpu line?", expected_route="web"),
        GateCase("web_inflation", "Current inflation trends in the US.", expected_route="web"),
        GateCase(
            "web_forced_url",
            "What is the capital of Italy?",
            expected_route="web",
            url="https://en.wikipedia.org/wiki/Italy",
            answer_must_contain="rome",
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate confidence gate routing (direct vs web)")
    ap.add_argument("--backend", default="hf", choices=["hf", "tiny"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tiny_ckpt", default="checkpoints_v2/final.pt")
    ap.add_argument("--tiny_tokenizer", default="tokenizer.model")
    ap.add_argument("--tiny_lora", default="")
    ap.add_argument("--tiny_top_p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--search_results", type=int, default=5)
    ap.add_argument("--timeout_sec", type=int, default=20)
    ap.add_argument("--direct_confidence_threshold", type=float, default=0.72)
    ap.add_argument("--out_json", default="data/eval_confidence_gate.json")
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
        search_results=int(args.search_results),
        timeout_sec=int(args.timeout_sec),
        direct_confidence_threshold=float(args.direct_confidence_threshold),
    )
    assistant = GroundedWebAssistant(cfg)

    results: List[GateResult] = []
    passed = 0
    cases = default_cases()
    for c in cases:
        out = assistant.answer(c.question, url=c.url, search_if_missing=(not bool(c.url)))
        route = str(out.debug.get("route", "unknown"))
        ok = route == c.expected_route
        if ok and c.answer_must_contain:
            ok = c.answer_must_contain.lower() in out.answer.lower()
        passed += int(ok)
        results.append(
            GateResult(
                label=c.label,
                passed=ok,
                expected_route=c.expected_route,
                route=route,
                answer=out.answer,
                sources=out.sources,
                debug=out.debug,
            )
        )
        print(f"[{'OK' if ok else 'FAIL'}] {c.label} -> route={route}")

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
