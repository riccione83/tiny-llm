#!/usr/bin/env python3
"""
Evaluate a code assistant model on programming + code review prompts.

Supports:
- base model only
- base model + LoRA adapter
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str) -> torch.dtype:
    n = (name or "auto").strip().lower()
    if n == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if n in {"float16", "fp16"}:
        return torch.float16
    if n in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if n in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_prompts(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
            rows.append(obj)
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = temperature > 1e-6
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max(16, int(max_new_tokens)),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def _has_any_regex(text: str, patterns: List[str]) -> bool:
    for p in patterns:
        try:
            if re.search(p, text):
                return True
        except re.error:
            continue
    return False


def score_output(task: Dict[str, object], output: str) -> Tuple[float, Dict[str, float]]:
    low = (output or "").lower()
    must = [str(x).lower() for x in (task.get("must_contain") or []) if str(x).strip()]
    regexes = [str(x) for x in (task.get("preferred_regex") or []) if str(x).strip()]
    task_type = str(task.get("task_type", "")).strip().lower()

    metrics: Dict[str, float] = {}
    metrics["non_empty"] = 1.0 if output.strip() else 0.0

    if must:
        hits = sum(1 for m in must if m in low)
        metrics["must_contain"] = float(hits) / float(len(must))
    else:
        metrics["must_contain"] = 1.0

    metrics["regex_hint"] = 1.0 if (not regexes or _has_any_regex(output, regexes)) else 0.0

    if task_type == "review":
        metrics["review_structure"] = 1.0 if ("finding" in low or "severity" in low) else 0.0
        metrics["test_orientation"] = 1.0 if ("test" in low or "pytest" in low) else 0.0
    elif task_type == "generate":
        metrics["code_block"] = 1.0 if "```" in output else 0.0
    elif task_type == "refactor":
        metrics["refactor_signal"] = 1.0 if ("refactor" in low or "improv" in low or "fix" in low) else 0.5

    score = float(sum(metrics.values()) / max(1, len(metrics)))
    return score, metrics


def aggregate_by_task_type(rows: List[Dict[str, object]], pass_threshold: float) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[float]] = {}
    for r in rows:
        t = str(r.get("task_type", "unknown")).strip().lower() or "unknown"
        groups.setdefault(t, []).append(float(r.get("score", 0.0)))

    out: Dict[str, Dict[str, float]] = {}
    for t, vals in groups.items():
        passed = sum(1 for v in vals if v >= pass_threshold)
        out[t] = {
            "count": float(len(vals)),
            "avg_score": float(sum(vals) / max(1, len(vals))),
            "pass_rate": float(passed) / float(len(vals)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate code-assistant behavior on fixed prompts")
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--adapter_dir", default="")
    ap.add_argument("--prompts_jsonl", default="samples/eval/code_assistant_eval.jsonl")
    ap.add_argument("--out_json", default="models/code_assistant_eval_report.json")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--pass_threshold", type=float, default=0.72)
    ap.add_argument("--device_map", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    base_model_dir = Path(args.base_model_dir)
    prompts_path = Path(args.prompts_jsonl)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path)
    dtype = resolve_dtype(args.dtype)

    tok = AutoTokenizer.from_pretrained(str(base_model_dir), use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    load_kwargs = {"torch_dtype": dtype}
    if str(args.device_map) == "auto":
        load_kwargs["device_map"] = "auto"
    elif str(args.device_map) == "cpu":
        load_kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(str(base_model_dir), **load_kwargs)
    if str(args.device_map) == "cuda":
        model = model.to("cuda")

    if str(args.adapter_dir).strip():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(Path(args.adapter_dir)))

    model.eval()

    rows: List[Dict[str, object]] = []
    all_scores: List[float] = []

    for task in prompts:
        prompt = str(task.get("prompt", ""))
        out = generate_one(
            model=model,
            tokenizer=tok,
            prompt=prompt,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
        score, metrics = score_output(task, out)
        all_scores.append(score)
        task_id = str(task.get("id", "unknown"))
        print(f"[{task_id}] score={score:.3f}")
        print(f"out: {out[:220]}{'...' if len(out) > 220 else ''}\n")
        rows.append(
            {
                "id": task_id,
                "task_type": str(task.get("task_type", "")),
                "score": score,
                "metrics": metrics,
                "prompt": prompt,
                "output": out,
            }
        )

    overall = float(sum(all_scores) / max(1, len(all_scores)))
    pass_threshold = float(args.pass_threshold)
    pass_count = sum(1 for s in all_scores if s >= pass_threshold)
    by_task_type = aggregate_by_task_type(rows, pass_threshold=pass_threshold)
    report = {
        "base_model_dir": str(base_model_dir.resolve()),
        "adapter_dir": str(Path(args.adapter_dir).resolve()) if str(args.adapter_dir).strip() else "",
        "prompts_jsonl": str(prompts_path.resolve()),
        "overall_score": overall,
        "pass_threshold": pass_threshold,
        "pass_count": int(pass_count),
        "pass_rate": float(pass_count) / max(1, len(rows)),
        "by_task_type": by_task_type,
        "count": len(rows),
        "results": rows,
    }
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Overall score: {overall:.3f}")
    print(f"Saved report: {out_json.resolve()}")


if __name__ == "__main__":
    main()
