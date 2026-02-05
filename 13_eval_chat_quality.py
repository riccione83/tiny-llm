#!/usr/bin/env python3
"""
Quick regression evaluator for 09_chat.py.

Runs a fixed prompt suite and reports pass/fail checks for:
- constraints (math / yes-no)
- summarization
- web-assisted factual queries
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple


def _run_chat(
    python_exe: str,
    chat_script: str,
    base_ckpt: str,
    tokenizer: str,
    lora_adapter: str,
    prompts: List[str],
    web_results: int,
) -> str:
    payload = "\n".join(prompts + ["/exit"]) + "\n"
    cmd = [
        python_exe,
        chat_script,
        "--base_ckpt",
        base_ckpt,
        "--tokenizer",
        tokenizer,
        "--temperature",
        "0.0",
        "--top_p",
        "1.0",
        "--confidence_threshold",
        "0.18",
        "--web_results",
        str(web_results),
    ]
    if lora_adapter:
        cmd += ["--lora_adapter", lora_adapter]
    p = subprocess.run(cmd, input=payload, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"chat process failed ({p.returncode})\nSTDERR:\n{p.stderr}")
    return p.stdout


def _extract_answers(stdout: str) -> List[str]:
    # In piped mode, each turn is printed as: "You: Bot: <answer>\n"
    chunks = stdout.split("You: ")
    out: List[str] = []
    for ch in chunks:
        s = ch.strip()
        if not s.startswith("Bot:"):
            continue
        ans = s[len("Bot:") :].strip()
        out.append(ans)
    return out


def _source_count(answer: str) -> int:
    if "Sources:" not in answer and "Source:" not in answer:
        return 0
    return len(re.findall(r"^\s*-\s+https?://", answer, flags=re.MULTILINE))


def _is_number_only(answer: str) -> bool:
    return re.fullmatch(r"-?\d+(?:\.\d+)?", answer.strip()) is not None


def _is_yes_no_only(answer: str) -> bool:
    return answer.strip().upper() in {"YES", "NO"}


def _two_sentence_like(answer: str) -> bool:
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    parts = [p for p in parts if p.strip()]
    return len(parts) >= 2


def _contains_any(text: str, needles: List[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _inflation_answer_ok(answer: str) -> bool:
    low = answer.lower()
    has_sources = _source_count(answer) >= 2
    has_uk = "uk" in low or "united kingdom" in low
    has_us = re.search(r"\bus\b", low) is not None or "united states" in low
    has_inflation = "inflation" in low or "cpi" in low
    has_data_signal = bool(re.search(r"\b\d+(?:\.\d+)?%?\b", low)) or ("trend" in low) or ("rose" in low) or ("fell" in low)
    return has_sources and has_uk and has_us and has_inflation and has_data_signal


def _default_suite() -> List[Tuple[str, Callable[[str], bool], str]]:
    return [
        ("What is 81/9? Answer with a number only.", _is_number_only, "number-only constraint"),
        ("Reply YES or NO only: Is Rome in Italy?", _is_yes_no_only, "YES/NO-only constraint"),
        (
            "Summarize in 2 sentences: She was studying alone but suddenly her friend knocked at the door. "
            "She got distracted and finished only half of her homework.",
            _two_sentence_like,
            "2-sentence summarization",
        ),
        (
            "What are the latest Nvidia GPUs and their specs?",
            lambda a: _source_count(a) >= 2 and _contains_any(a, ["spec", "rtx", "geforce", "memory", "cuda"]),
            "web specs answer with sources",
        ),
        (
            "How do NVIDIA Ada Lovelace and Blackwell architectures differ?",
            lambda a: _source_count(a) >= 2 and _contains_any(a, ["compare", "difference", "blackwell", "ada"]),
            "compare answer with sources",
        ),
        (
            "What are the latest developments in AI regulation in EU?",
            lambda a: _source_count(a) >= 2 and _contains_any(a, ["ai act", "regulation", "european", "eu"]),
            "EU regulation answer with sources",
        ),
        (
            "What are the most recent SpaceX launches and outcomes?",
            lambda a: _source_count(a) >= 1 and _contains_any(a, ["spacex", "launch", "outcome", "success", "failed"]),
            "SpaceX launches/outcomes answer",
        ),
        (
            "Compare the latest AMD Radeon GPUs vs Nvidia GPUs.",
            lambda a: _source_count(a) >= 2 and _contains_any(a, ["compare", "amd", "nvidia", "radeon", "rtx"]),
            "AMD vs Nvidia compare answer",
        ),
        (
            "Current inflation trends in the UK and US.",
            _inflation_answer_ok,
            "UK/US inflation answer",
        ),
        (
            "Best travel tips for visiting Tokyo this year.",
            lambda a: _source_count(a) >= 2 and _contains_any(a, ["tokyo", "travel", "tips", "visit"]),
            "Tokyo travel answer",
        ),
        (
            "Latest stock performance and news about Nvidia (NVDA).",
            lambda a: _source_count(a) >= 1 and _contains_any(a, ["nvda", "close", "change", "stock"]),
            "NVDA stock snapshot answer",
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--chat_script", default="09_chat.py")
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lora_adapter", default="")
    ap.add_argument("--web_results", type=int, default=3)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    suite = _default_suite()
    prompts = [x[0] for x in suite]
    stdout = _run_chat(
        python_exe=args.python,
        chat_script=args.chat_script,
        base_ckpt=args.base_ckpt,
        tokenizer=args.tokenizer,
        lora_adapter=args.lora_adapter,
        prompts=prompts,
        web_results=args.web_results,
    )
    answers = _extract_answers(stdout)

    results: List[Dict[str, object]] = []
    passed = 0
    for i, (prompt, check_fn, label) in enumerate(suite):
        ans = answers[i] if i < len(answers) else ""
        ok = bool(check_fn(ans))
        passed += int(ok)
        results.append(
            {
                "index": i + 1,
                "label": label,
                "prompt": prompt,
                "pass": ok,
                "answer": ans,
            }
        )

    total = len(suite)
    print(f"PASS {passed}/{total}")
    for r in results:
        status = "OK" if r["pass"] else "FAIL"
        print(f"[{status}] {r['index']}. {r['label']}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"passed": passed, "total": total, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved report: {out_path}")

    # non-zero exit if any failure (useful in CI/local checks)
    if passed < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
