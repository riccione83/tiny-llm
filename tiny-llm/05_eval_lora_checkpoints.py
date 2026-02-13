#!/usr/bin/env python3
"""
Compare LoRA checkpoints on a fixed prompt set.

Loads the base model once, then applies each adapter checkpoint and runs
generation on the same prompts so you can inspect progression during training.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "User: Scrivi un messaggio di benvenuto in italiano, amichevole ma professionale (2 frasi).\n\nAssistant:",
    "User: Explain in a short sentence why a cat is a mammal.\n\nAssistant:",
    "User: Write a short checklist (5 items) for debugging a Python bug.\n\nAssistant:",
    "User: Risolvi: 18*7. Rispondi con numero + una frase breve.\n\nAssistant:",
    "User: Se tutti i gatti sono mammiferi e tutti i mammiferi respirano, i gatti respirano? Rispondi si/no + 1 riga.\n\nAssistant:",
]
ITALIAN_HINTS = {
    "ciao",
    "benvenuto",
    "benvenuta",
    "giornata",
    "grazie",
    "aiutarti",
    "italiano",
    "sono",
    "perche",
    "gatti",
    "mammiferi",
    "respirano",
    "si",
    "buona",
}
ENGLISH_HINTS = {"the", "and", "for", "with", "use", "check", "code", "debug", "python", "is", "a", "to"}


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


def list_checkpoints(adapter_dir: Path, checkpoint: str, limit: int) -> List[Path]:
    if checkpoint:
        p = Path(checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return [p]

    found = [p for p in adapter_dir.glob("checkpoint-*") if p.is_dir()]
    found.sort(key=lambda p: int(re.sub(r"^checkpoint-", "", p.name)))
    if limit > 0 and len(found) > limit:
        found = found[-limit:]
    return found


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = temperature > 1e-6
    if not do_sample:
        # Avoid spurious warnings on some model generation configs.
        try:
            model.generation_config.top_k = None
            model.generation_config.top_p = None
            model.generation_config.temperature = None
        except Exception:
            pass
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


def _split_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ']+", (text or "").lower())


def _is_repetitive(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # crude loop detector: repeated 4-word chunks.
    words = _split_words(t)
    if len(words) < 12:
        return False
    chunks = [" ".join(words[i : i + 4]) for i in range(0, len(words) - 3)]
    uniq = set(chunks)
    return (len(uniq) / max(1, len(chunks))) < 0.55


def _has_garbage_pattern(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # Long same-char run or long digit run are strong degeneration signals.
    if re.search(r"(.)\1{7,}", t):
        return True
    if re.search(r"\d{6,}", t):
        return True
    return False


def _language_score(text: str, expected: str) -> float:
    words = _split_words(text)
    if not words:
        return 0.0
    it_hits = sum(1 for w in words if w in ITALIAN_HINTS)
    en_hits = sum(1 for w in words if w in ENGLISH_HINTS)
    if expected == "it":
        return min(1.0, (it_hits + 1) / (en_hits + 1))
    if expected == "en":
        return min(1.0, (en_hits + 1) / (it_hits + 1))
    return 0.5


def score_sample(prompt: str, output: str, idx: int) -> Tuple[float, Dict[str, float]]:
    """
    Returns a [0..1] score and sub-metrics.
    idx uses DEFAULT_PROMPTS semantics when prompts are default.
    """
    out = (output or "").strip()
    lines = [x for x in out.splitlines() if x.strip()]
    metrics: Dict[str, float] = {
        "non_empty": 1.0 if out else 0.0,
        "not_repetitive": 0.0 if _is_repetitive(out) else 1.0,
        "no_garbage": 0.0 if _has_garbage_pattern(out) else 1.0,
    }
    if idx == 1:  # Italian welcome
        # prefer 1-3 short sentences, IT language, no aggressive question spam.
        sents = [s for s in re.split(r"[.!?]+", out) if s.strip()]
        q_marks = out.count("?")
        words = _split_words(out)
        low = out.lower()
        metrics["lang"] = _language_score(out, "it")
        metrics["length"] = 1.0 if 1 <= len(sents) <= 3 else 0.4
        metrics["question_penalty"] = 1.0 if q_marks <= 1 else 0.3
        metrics["focus_welcome"] = 1.0 if any(x in low for x in ["benvenut", "ciao", "buon giorno", "buona giornata"]) else 0.4
        metrics["concise_words"] = 1.0 if 8 <= len(words) <= 40 else 0.4
        metrics["avoid_bad_pattern"] = 0.2 if "buona notte" in low else 1.0
    elif idx == 2:  # short EN fact
        sents = [s for s in re.split(r"[.!?]+", out) if s.strip()]
        low = out.lower()
        metrics["lang"] = _language_score(out, "en")
        metrics["length"] = 1.0 if len(sents) <= 2 else 0.4
        metrics["mentions_cat_mammal"] = 1.0 if ("cat" in out.lower() and "mammal" in out.lower()) else 0.2
        metrics["factual_hint"] = 0.2 if "lay eggs" in low else 1.0
    elif idx == 3:  # checklist EN
        bullet_lines = sum(1 for x in lines if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", x))
        metrics["lang"] = _language_score(out, "en")
        metrics["bullets"] = 1.0 if bullet_lines >= 4 else (0.6 if bullet_lines >= 2 else 0.2)
    elif idx == 4:  # 18*7
        has_126 = "126" in out
        sents = [s for s in re.split(r"[.!?]+", out) if s.strip()]
        low = out.lower()
        metrics["correct"] = 1.0 if has_126 else 0.0
        metrics["concise"] = 1.0 if len(sents) <= 3 else 0.4
        metrics["avoid_wrong_expl"] = 0.2 if ("somma di 18 e 7" in low or "suma de 18 y 7" in low) else 1.0
    elif idx == 5:  # logic yes/no italian
        low = out.lower()
        yesno = any(x in low for x in ["si", "sì", "yes", "no"])
        words = _split_words(out)
        metrics["lang"] = _language_score(out, "it")
        metrics["yes_no"] = 1.0 if yesno else 0.2
        metrics["mentions_resp"] = 1.0 if any(x in low for x in ["respir", "mammifer", "gatti", "cats"]) else 0.3
        metrics["not_too_short"] = 1.0 if len(words) >= 4 else 0.4
        metrics["not_too_long"] = 1.0 if len(words) <= 40 else 0.5
    else:
        metrics["generic_len"] = 1.0 if len(lines) <= 8 else 0.5
        metrics["generic_not_too_long"] = 1.0 if len(_split_words(out)) <= 120 else 0.5

    score = sum(metrics.values()) / max(1, len(metrics))
    return float(score), metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate LoRA checkpoints on fixed prompts")
    ap.add_argument("--base_model_dir", default="models/base_trained")
    ap.add_argument("--adapter_dir", default="models/lora_adapter")
    ap.add_argument("--checkpoint", default="", help="Single checkpoint path to evaluate")
    ap.add_argument("--max_checkpoints", type=int, default=5, help="Newest N checkpoints when --checkpoint is not set")
    ap.add_argument("--prompt", action="append", default=[], help="Extra prompt(s); if omitted uses built-ins")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument(
        "--device_map",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Model placement mode for eval. Use 'cuda' to force full GPU load when possible.",
    )
    ap.add_argument(
        "--offload_dir",
        default="",
        help="Disk offload directory used when device_map='auto' spills layers to CPU/disk.",
    )
    ap.add_argument("--out_json", default="models/lora_adapter/checkpoint_eval_report.json")
    args = ap.parse_args()

    base_model_dir = Path(args.base_model_dir)
    adapter_dir = Path(args.adapter_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    prompts = [p.strip() for p in args.prompt if p.strip()] or list(DEFAULT_PROMPTS)
    checkpoints = list_checkpoints(adapter_dir, args.checkpoint, int(args.max_checkpoints))
    if not checkpoints:
        raise SystemExit("No checkpoints found to evaluate.")

    print(f"Loading tokenizer from: {base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), use_fast=True)
    dtype = resolve_dtype(args.dtype)
    offload_dir = Path(args.offload_dir) if str(args.offload_dir).strip() else (adapter_dir / "_eval_offload")
    offload_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {
        "base_model_dir": str(base_model_dir.resolve()),
        "adapter_dir": str(adapter_dir.resolve()),
        "prompts": prompts,
        "results": [],
    }
    ckpt_scores: List[Tuple[str, float]] = []

    for ckpt in checkpoints:
        print(f"\n=== {ckpt.name} ===")
        # Load a fresh base model for each checkpoint to avoid stacking adapters.
        load_kwargs = {
            "torch_dtype": dtype,
        }
        if str(args.device_map) == "auto":
            load_kwargs["device_map"] = "auto"
            load_kwargs["offload_folder"] = str(offload_dir)
            load_kwargs["offload_state_dict"] = True
        elif str(args.device_map) == "cpu":
            load_kwargs["device_map"] = {"": "cpu"}
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model_dir),
            **load_kwargs,
        )
        if str(args.device_map) == "cuda":
            model = model.to("cuda")
        model.eval()
        peft_kwargs = {}
        if str(args.device_map) == "auto":
            peft_kwargs["offload_dir"] = str(offload_dir)
        peft_model = PeftModel.from_pretrained(model, str(ckpt), **peft_kwargs)
        peft_model.eval()
        rows = []
        sample_scores: List[float] = []
        for idx, prompt in enumerate(prompts, start=1):
            out = generate_one(
                model=peft_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            s, sm = score_sample(prompt=prompt, output=out, idx=idx)
            print(f"[{idx}] prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            print(f"    out: {out[:240]}{'...' if len(out) > 240 else ''}")
            print(f"    score: {s:.3f}")
            rows.append({"prompt": prompt, "output": out, "score": s, "subscores": sm})
            sample_scores.append(s)

        ckpt_score = float(sum(sample_scores) / max(1, len(sample_scores)))
        ckpt_scores.append((ckpt.name, ckpt_score))
        print(f"Checkpoint score: {ckpt_score:.3f}")

        report["results"].append(
            {
                "checkpoint": str(ckpt.resolve()),
                "checkpoint_score": ckpt_score,
                "samples": rows,
            }
        )
        # release adapter weights before next checkpoint
        del peft_model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if ckpt_scores:
        ranked = sorted(ckpt_scores, key=lambda x: x[1], reverse=True)
        print("\n=== Scoreboard ===")
        for rank, (name, score) in enumerate(ranked, start=1):
            print(f"{rank}. {name}: {score:.3f}")
    print(f"\nSaved report: {out_json.resolve()}")


if __name__ == "__main__":
    main()
