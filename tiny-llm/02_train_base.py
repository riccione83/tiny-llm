#!/usr/bin/env python3
"""
Modern base training pipeline (continued pretraining / domain adaptation).

Design goals:
- Stream large datasets from Hugging Face without full local download.
- Mix multiple sources (knowledge-first defaults).
- Keep a single script for long multi-day training runs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


@dataclass
class HFSource:
    name: str
    config: Optional[str]
    split: str
    text_field: Optional[str]
    max_texts: int


RECIPE_SOURCES: Dict[str, List[HFSource]] = {
    "tiny": [
        HFSource("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 40_000),
        HFSource("allenai/c4", "en", "train", "text", 30_000),
    ],
    "standard": [
        HFSource("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 300_000),
        HFSource("allenai/c4", "en", "train", "text", 250_000),
        HFSource("HuggingFaceFW/fineweb", "sample-10BT", "train", "text", 120_000),
    ],
    "knowledge-heavy": [
        HFSource("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 1_200_000),
        HFSource("allenai/c4", "en", "train", "text", 1_000_000),
        HFSource("HuggingFaceFW/fineweb", "sample-10BT", "train", "text", 500_000),
    ],
}


def normalize_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


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


def parse_hf_source(spec: str, fallback_max_texts: int) -> HFSource:
    # Format: name|config|split|text_field|max_texts
    parts = (spec or "").split("|")
    parts += [""] * (5 - len(parts))
    name = parts[0].strip()
    config = parts[1].strip() or None
    split = (parts[2].strip() or "train")
    text_field = parts[3].strip() or None
    max_texts_str = parts[4].strip()
    max_texts = int(max_texts_str) if max_texts_str else int(fallback_max_texts)
    if not name:
        raise ValueError(f"Invalid hf source spec: {spec}")
    return HFSource(name=name, config=config, split=split, text_field=text_field, max_texts=max_texts)


def row_to_text(row: Dict[str, object], text_field: Optional[str]) -> Optional[str]:
    if text_field and isinstance(row.get(text_field), str):
        return normalize_text(str(row.get(text_field)))

    if isinstance(row.get("text"), str):
        return normalize_text(str(row["text"]))

    if isinstance(row.get("title"), str) and isinstance(row.get("text"), str):
        return normalize_text(f"{row['title']}\n\n{row['text']}")

    messages = row.get("messages")
    if isinstance(messages, list):
        chunks: List[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user")).strip().capitalize()
            content = normalize_text(str(m.get("content", "")))
            if content:
                chunks.append(f"{role}: {content}")
        if chunks:
            return "\n\n".join(chunks)

    if isinstance(row.get("question"), str) and isinstance(row.get("response"), str):
        sys_prompt = normalize_text(str(row.get("system_prompt", "")))
        q = normalize_text(str(row["question"]))
        a = normalize_text(str(row["response"]))
        pieces = []
        if sys_prompt:
            pieces.append(f"System: {sys_prompt}")
        pieces.append(f"User: {q}")
        pieces.append(f"Assistant: {a}")
        return "\n\n".join(pieces)

    if isinstance(row.get("instruction"), str) and isinstance(row.get("output"), str):
        instruction = normalize_text(str(row["instruction"]))
        extra = normalize_text(str(row.get("input", "")))
        output = normalize_text(str(row["output"]))
        prompt = instruction if not extra else f"{instruction}\n\nInput: {extra}"
        return f"User: {prompt}\n\nAssistant: {output}"

    return None


def make_hf_text_iter(
    source: HFSource,
    allow_remote_dataset_code: bool,
    min_chars: int,
) -> Callable[[], Iterator[str]]:
    def _iter() -> Iterator[str]:
        kwargs = {"split": source.split, "streaming": True, "trust_remote_code": allow_remote_dataset_code}
        if source.config:
            ds = load_dataset(source.name, source.config, **kwargs)
        else:
            ds = load_dataset(source.name, **kwargs)

        count = 0
        for row in ds:
            txt = row_to_text(row, source.text_field)
            if not txt or len(txt) < min_chars:
                continue
            yield txt
            count += 1
            if source.max_texts > 0 and count >= source.max_texts:
                break

    return _iter


def make_local_text_iter(glob_pattern: str, min_chars: int) -> Callable[[], Iterator[str]]:
    def _iter() -> Iterator[str]:
        for p in sorted(Path(".").glob(glob_pattern)):
            if not p.is_file():
                continue
            try:
                txt = normalize_text(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            if txt and len(txt) >= min_chars:
                yield txt

    return _iter


def make_local_jsonl_iter(glob_pattern: str, text_key: str, min_chars: int) -> Callable[[], Iterator[str]]:
    def _iter() -> Iterator[str]:
        for p in sorted(Path(".").glob(glob_pattern)):
            if not p.is_file():
                continue
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        txt = None
                        if isinstance(row, dict):
                            if text_key and isinstance(row.get(text_key), str):
                                txt = normalize_text(str(row.get(text_key)))
                            else:
                                txt = row_to_text(row, None)
                        if txt and len(txt) >= min_chars:
                            yield txt
            except Exception:
                continue

    return _iter


def make_round_robin_text_iter(
    sources: List[Tuple[str, Callable[[], Iterator[str]]]],
    repeat: bool,
    seed: int,
) -> Callable[[], Iterator[str]]:
    def _iter() -> Iterator[str]:
        cycle = 0
        while True:
            if not sources:
                return
            idxs = list(range(len(sources)))
            rnd = random.Random(seed + cycle)
            rnd.shuffle(idxs)
            active: List[Tuple[str, Iterator[str]]] = []
            for i in idxs:
                name, fn = sources[i]
                try:
                    active.append((name, iter(fn())))
                except Exception:
                    continue
            if not active:
                return
            while active:
                next_active: List[Tuple[str, Iterator[str]]] = []
                for name, it in active:
                    try:
                        yield next(it)
                        next_active.append((name, it))
                    except StopIteration:
                        pass
                    except Exception:
                        pass
                active = next_active
            if not repeat:
                break
            cycle += 1

    return _iter


class PackedCausalDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        text_iter_factory: Callable[[], Iterator[str]],
        block_size: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.text_iter_factory = text_iter_factory
        self.block_size = int(block_size)

    def __iter__(self):
        eos = self.tokenizer.eos_token_id
        buffer: List[int] = []
        for text in self.text_iter_factory():
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            if eos is not None:
                ids.append(int(eos))
            buffer.extend(ids)
            while len(buffer) >= self.block_size:
                chunk = buffer[: self.block_size]
                del buffer[: self.block_size]
                x = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids": x,
                    "attention_mask": torch.ones_like(x),
                    "labels": x.clone(),
                }


def save_interrupt_checkpoint(trainer: Trainer, tokenizer: AutoTokenizer, out_dir: Path, reason: str) -> Path:
    step = int(getattr(trainer.state, "global_step", 0))
    if step > 0 and hasattr(trainer, "_save_checkpoint"):
        try:
            trainer._save_checkpoint(trainer.model, trial=None)  # Uses Trainer checkpoint layout for resume support.
            ckpt = out_dir / f"checkpoint-{step}"
            if ckpt.exists():
                tokenizer.save_pretrained(str(ckpt))
                print(f"Saved interrupt checkpoint ({reason}) at: {ckpt.resolve()}")
                return ckpt
        except Exception as exc:
            print(f"Warning: failed to write full trainer checkpoint on interrupt: {exc}")

    fallback = out_dir / f"interrupt-step-{step}"
    fallback.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(fallback))
    tokenizer.save_pretrained(str(fallback))
    try:
        trainer.state.save_to_json(str(fallback / "trainer_state.json"))
    except Exception:
        pass
    print(f"Saved fallback interrupt checkpoint ({reason}) at: {fallback.resolve()}")
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser(description="Train base model with streaming knowledge corpora")
    ap.add_argument("--model_dir", default="models/base")
    ap.add_argument("--output_dir", default="models/base_trained")
    ap.add_argument("--recipe", default="standard", choices=["tiny", "standard", "knowledge-heavy"])
    ap.add_argument("--hf_source", action="append", default=[], help="Extra source: name|config|split|text_field|max_texts")
    ap.add_argument("--disable_hf_data", action="store_true", help="Use only local data globs")
    ap.add_argument("--allow_remote_dataset_code", action="store_true")
    ap.add_argument(
        "--local_text_glob",
        action="append",
        default=["samples/base/**/*.txt", "tiny-llm/samples/base/**/*.txt"],
    )
    ap.add_argument(
        "--local_jsonl_glob",
        action="append",
        default=["samples/base/**/*.jsonl", "tiny-llm/samples/base/**/*.jsonl"],
    )
    ap.add_argument("--jsonl_text_key", default="text")
    ap.add_argument("--disable_local_data", action="store_true")
    ap.add_argument("--min_chars", type=int, default=80)
    ap.add_argument("--max_texts_per_source", type=int, default=0, help="0 = use recipe defaults")
    ap.add_argument("--repeat_sources", action="store_true")

    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=30_000)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--resume_from_checkpoint", default="")
    args = ap.parse_args()

    set_seed(int(args.seed))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.config.use_cache = False

    if tokenizer.vocab_size > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(tokenizer.vocab_size)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    sources: List[Tuple[str, Callable[[], Iterator[str]]]] = []
    if not args.disable_hf_data:
        for src in RECIPE_SOURCES[args.recipe]:
            max_texts = int(args.max_texts_per_source) if int(args.max_texts_per_source) > 0 else int(src.max_texts)
            resolved = HFSource(
                name=src.name,
                config=src.config,
                split=src.split,
                text_field=src.text_field,
                max_texts=max_texts,
            )
            fn = make_hf_text_iter(
                source=resolved,
                allow_remote_dataset_code=bool(args.allow_remote_dataset_code),
                min_chars=int(args.min_chars),
            )
            sources.append((f"hf:{resolved.name}", fn))

        for spec in args.hf_source:
            resolved = parse_hf_source(spec, fallback_max_texts=int(args.max_texts_per_source))
            fn = make_hf_text_iter(
                source=resolved,
                allow_remote_dataset_code=bool(args.allow_remote_dataset_code),
                min_chars=int(args.min_chars),
            )
            sources.append((f"hf:{resolved.name}", fn))

    if not args.disable_local_data:
        for g in args.local_text_glob:
            sources.append((f"local_txt:{g}", make_local_text_iter(g, int(args.min_chars))))
        for g in args.local_jsonl_glob:
            sources.append(
                (
                    f"local_jsonl:{g}",
                    make_local_jsonl_iter(g, text_key=args.jsonl_text_key, min_chars=int(args.min_chars)),
                )
            )

    if not sources:
        raise SystemExit("No training sources found. Provide --hf_source or local data globs.")

    text_iter_factory = make_round_robin_text_iter(
        sources=sources,
        repeat=bool(args.repeat_sources),
        seed=int(args.seed),
    )
    dataset = PackedCausalDataset(
        tokenizer=tokenizer,
        text_iter_factory=text_iter_factory,
        block_size=int(args.block_size),
    )

    use_bf16 = dtype == torch.bfloat16 and torch.cuda.is_available()
    use_fp16 = dtype == torch.float16 and torch.cuda.is_available()

    targs = TrainingArguments(
        output_dir=str(out),
        overwrite_output_dir=True,
        save_strategy="steps",
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.per_device_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        lr_scheduler_type=str(args.lr_scheduler_type),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        bf16=bool(use_bf16),
        fp16=bool(use_fp16),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim="adamw_torch",
        save_safetensors=True,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM) if hasattr(signal, "SIGTERM") else None
    signal_name = {"value": "KeyboardInterrupt"}

    def _interrupt_handler(signum, _frame):
        try:
            signal_name["value"] = signal.Signals(signum).name
        except Exception:
            signal_name["value"] = f"signal-{signum}"
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _interrupt_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _interrupt_handler)

    interrupted = False
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    except KeyboardInterrupt:
        interrupted = True
        save_interrupt_checkpoint(trainer, tokenizer, out, signal_name["value"])
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)

    if interrupted:
        return

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    run_meta = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "output_dir": str(out.resolve()),
        "recipe": args.recipe,
        "sources": [name for name, _ in sources],
        "block_size": int(args.block_size),
        "max_steps": int(args.max_steps),
        "learning_rate": float(args.learning_rate),
        "dtype": str(dtype),
        "repeat_sources": bool(args.repeat_sources),
    }
    (out / "base_training_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("Base training complete.")
    print(f"- output_dir: {out.resolve()}")


if __name__ == "__main__":
    main()
