#!/usr/bin/env python3
"""
Compare two code-assistant eval reports produced by 08_eval_code_assistant.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_report(path: Path) -> Dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid report format: {path}")
    return obj


def _index_results(report: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    idx: Dict[str, Dict[str, object]] = {}
    for row in report.get("results", []) or []:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id", "")).strip()
        if rid:
            idx[rid] = row
    return idx


def _task_type_avg(report: Dict[str, object]) -> Dict[str, float]:
    by_type = report.get("by_task_type")
    if isinstance(by_type, dict):
        out: Dict[str, float] = {}
        for k, v in by_type.items():
            if isinstance(v, dict):
                out[str(k)] = float(v.get("avg_score", 0.0))
        if out:
            return out

    # Fallback for older reports
    sums: Dict[str, Tuple[float, int]] = {}
    for row in report.get("results", []) or []:
        if not isinstance(row, dict):
            continue
        t = str(row.get("task_type", "unknown")).strip().lower() or "unknown"
        s = float(row.get("score", 0.0))
        old_sum, old_n = sums.get(t, (0.0, 0))
        sums[t] = (old_sum + s, old_n + 1)
    return {k: (v[0] / max(1, v[1])) for k, v in sums.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline/candidate code-assistant eval reports")
    ap.add_argument("--baseline_report", required=True)
    ap.add_argument("--candidate_report", required=True)
    ap.add_argument("--out_json", default="models/code_assistant_compare_report.json")
    args = ap.parse_args()

    baseline_path = Path(args.baseline_report)
    candidate_path = Path(args.candidate_report)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    base = load_report(baseline_path)
    cand = load_report(candidate_path)

    base_overall = float(base.get("overall_score", 0.0))
    cand_overall = float(cand.get("overall_score", 0.0))
    base_pass_rate = float(base.get("pass_rate", 0.0))
    cand_pass_rate = float(cand.get("pass_rate", 0.0))

    base_idx = _index_results(base)
    cand_idx = _index_results(cand)
    common_ids = sorted(set(base_idx.keys()) & set(cand_idx.keys()))

    per_task_delta: List[Dict[str, object]] = []
    for rid in common_ids:
        b = float(base_idx[rid].get("score", 0.0))
        c = float(cand_idx[rid].get("score", 0.0))
        per_task_delta.append(
            {
                "id": rid,
                "task_type": str(cand_idx[rid].get("task_type", "")),
                "baseline_score": b,
                "candidate_score": c,
                "delta": c - b,
            }
        )
    per_task_delta.sort(key=lambda x: float(x["delta"]), reverse=True)

    base_type = _task_type_avg(base)
    cand_type = _task_type_avg(cand)
    all_types = sorted(set(base_type.keys()) | set(cand_type.keys()))
    by_type_delta = [
        {
            "task_type": t,
            "baseline_avg": float(base_type.get(t, 0.0)),
            "candidate_avg": float(cand_type.get(t, 0.0)),
            "delta": float(cand_type.get(t, 0.0)) - float(base_type.get(t, 0.0)),
        }
        for t in all_types
    ]
    by_type_delta.sort(key=lambda x: float(x["delta"]), reverse=True)

    summary = {
        "baseline_report": str(baseline_path.resolve()),
        "candidate_report": str(candidate_path.resolve()),
        "overall": {
            "baseline_score": base_overall,
            "candidate_score": cand_overall,
            "delta": cand_overall - base_overall,
            "baseline_pass_rate": base_pass_rate,
            "candidate_pass_rate": cand_pass_rate,
            "pass_rate_delta": cand_pass_rate - base_pass_rate,
        },
        "counts": {
            "baseline_count": int(base.get("count", 0)),
            "candidate_count": int(cand.get("count", 0)),
            "common_ids": len(common_ids),
        },
        "by_task_type": by_type_delta,
        "top_gains": per_task_delta[:8],
        "top_losses": list(reversed(per_task_delta[-8:])),
    }

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    delta = float(summary["overall"]["delta"])
    pass_delta = float(summary["overall"]["pass_rate_delta"])
    print(f"Overall delta: {delta:+.4f}")
    print(f"Pass-rate delta: {pass_delta:+.4f}")
    print(f"Saved comparison: {out_json.resolve()}")


if __name__ == "__main__":
    main()
