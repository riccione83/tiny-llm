#!/usr/bin/env python3
"""
Modern LoRA training pipeline for instruction/chat alignment.

Uses PEFT + Transformers Trainer with streaming datasets.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import random
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


@dataclass
class HFSource:
    name: str
    config: Optional[str]
    split: str
    max_rows: int


RECIPE_SOURCES: Dict[str, List[HFSource]] = {
    "tiny": [
        HFSource("yahma/alpaca-cleaned", None, "train", 40_000),
        HFSource("databricks/databricks-dolly-15k", None, "train", 15_000),
    ],
    "standard": [
        HFSource("HuggingFaceH4/ultrachat_200k", None, "train_sft", 180_000),
        HFSource("Open-Orca/OpenOrca", None, "train", 300_000),
        HFSource("yahma/alpaca-cleaned", None, "train", 52_000),
        HFSource("databricks/databricks-dolly-15k", None, "train", 15_000),
    ],
    "heavy": [
        HFSource("HuggingFaceH4/ultrachat_200k", None, "train_sft", 207_000),
        HFSource("Open-Orca/OpenOrca", None, "train", 800_000),
        HFSource("yahma/alpaca-cleaned", None, "train", 52_000),
        HFSource("databricks/databricks-dolly-15k", None, "train", 15_000),
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


def resolve_attn_implementation(spec: str) -> Optional[str]:
    name = (spec or "auto").strip().lower()
    if name in {"", "none"}:
        return None
    if name == "auto":
        return "sdpa" if torch.cuda.is_available() else None
    return name


def configure_torch_runtime(enable_tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
    if enable_tf32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def resolve_optimizer_name(use_fused_optimizer: bool) -> str:
    base = "adamw_torch"
    if not use_fused_optimizer or (not torch.cuda.is_available()):
        return base
    try:
        from transformers.training_args import OptimizerNames

        available = {str(x.value) if hasattr(x, "value") else str(x) for x in OptimizerNames}
        if "adamw_torch_fused" not in available:
            return base
    except Exception:
        return base
    try:
        if "fused" not in inspect.signature(torch.optim.AdamW.__init__).parameters:
            return base
    except Exception:
        return base
    return "adamw_torch_fused"


def resolve_torch_compile_backend(spec: str) -> Optional[str]:
    name = (spec or "auto").strip().lower()
    if name in {"", "none"}:
        return None
    if name != "auto":
        return name
    if platform.system().lower().startswith("win"):
        return "aot_eager"
    return "inductor"


def load_causal_lm(
    model_id_or_path: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    attn_implementation: str,
):
    kwargs = {"trust_remote_code": bool(trust_remote_code)}
    attn_impl = resolve_attn_implementation(attn_implementation)
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            **kwargs,
        )


def make_training_arguments(**kwargs) -> TrainingArguments:
    supported = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered = {}
    dropped: List[str] = []
    for k, v in kwargs.items():
        if k in supported:
            filtered[k] = v
        else:
            dropped.append(k)
    if dropped:
        print(f"TrainingArguments compatibility: ignoring unsupported args: {', '.join(sorted(dropped))}")
    return TrainingArguments(**filtered)


def parse_hf_source(spec: str, fallback_max_rows: int) -> HFSource:
    # Format: name|config|split|max_rows
    parts = (spec or "").split("|")
    parts += [""] * (4 - len(parts))
    name = parts[0].strip()
    config = parts[1].strip() or None
    split = parts[2].strip() or "train"
    max_rows = int(parts[3]) if parts[3].strip() else int(fallback_max_rows)
    if not name:
        raise ValueError(f"Invalid hf source spec: {spec}")
    return HFSource(name=name, config=config, split=split, max_rows=max_rows)


def pairs_from_messages(messages: object) -> List[Tuple[str, str]]:
    if not isinstance(messages, list):
        return []
    cleaned: List[Tuple[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user")).strip().lower()
        content = normalize_text(str(m.get("content", "")))
        if not content:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"
        cleaned.append((role, content))
    if not cleaned:
        return []

    pairs: List[Tuple[str, str]] = []
    history_lines: List[str] = []
    for role, content in cleaned:
        if role == "assistant":
            prompt = "\n\n".join(history_lines + ["Assistant:"])
            if prompt.strip() and content.strip():
                pairs.append((prompt, content))
        prefix = "System" if role == "system" else ("User" if role == "user" else "Assistant")
        history_lines.append(f"{prefix}: {content}")
    return pairs


def row_to_pairs(row: Dict[str, object]) -> List[Tuple[str, str]]:
    pairs = pairs_from_messages(row.get("messages"))
    if pairs:
        return pairs

    if isinstance(row.get("question"), str) and isinstance(row.get("response"), str):
        sys_prompt = normalize_text(str(row.get("system_prompt", "")))
        question = normalize_text(str(row["question"]))
        answer = normalize_text(str(row["response"]))
        prompt_parts = []
        if sys_prompt:
            prompt_parts.append(f"System: {sys_prompt}")
        prompt_parts.append(f"User: {question}")
        prompt_parts.append("Assistant:")
        return [("\n\n".join(prompt_parts), answer)]

    if isinstance(row.get("instruction"), str):
        instruction = normalize_text(str(row.get("instruction", "")))
        inp = normalize_text(str(row.get("input", "")))
        ctx = normalize_text(str(row.get("context", "")))
        answer = normalize_text(str(row.get("output", row.get("response", ""))))
        if answer:
            user_text = instruction
            if inp:
                user_text = f"{user_text}\n\nInput: {inp}"
            if ctx:
                user_text = f"{user_text}\n\nContext: {ctx}"
            prompt = f"User: {user_text}\n\nAssistant:"
            return [(prompt, answer)]

    if isinstance(row.get("prompt"), str) and isinstance(row.get("response"), str):
        prompt = normalize_text(str(row["prompt"]))
        answer = normalize_text(str(row["response"]))
        return [(f"User: {prompt}\n\nAssistant:", answer)]

    return []


def make_hf_pair_iter(
    source: HFSource,
    allow_remote_dataset_code: bool,
) -> Callable[[], Iterator[Tuple[str, str]]]:
    def _iter() -> Iterator[Tuple[str, str]]:
        kwargs = {"split": source.split, "streaming": True, "trust_remote_code": allow_remote_dataset_code}
        if source.config:
            ds = load_dataset(source.name, source.config, **kwargs)
        else:
            ds = load_dataset(source.name, **kwargs)

        rows = 0
        for row in ds:
            rows += 1
            for prompt, answer in row_to_pairs(row):
                if prompt and answer:
                    yield (prompt, answer)
            if source.max_rows > 0 and rows >= source.max_rows:
                break

    return _iter


def make_local_jsonl_pair_iter(
    glob_pattern: str,
) -> Callable[[], Iterator[Tuple[str, str]]]:
    def _iter() -> Iterator[Tuple[str, str]]:
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
                        if not isinstance(row, dict):
                            continue
                        for prompt, answer in row_to_pairs(row):
                            if prompt and answer:
                                yield (prompt, answer)
            except Exception:
                continue

    return _iter


def make_round_robin_pair_iter(
    sources: List[Tuple[str, Callable[[], Iterator[Tuple[str, str]]]]],
    repeat: bool,
    seed: int,
) -> Callable[[], Iterator[Tuple[str, str]]]:
    def _iter() -> Iterator[Tuple[str, str]]:
        cycle = 0
        while True:
            if not sources:
                return
            idxs = list(range(len(sources)))
            rnd = random.Random(seed + cycle)
            rnd.shuffle(idxs)
            active: List[Tuple[str, Iterator[Tuple[str, str]]]] = []
            for i in idxs:
                name, fn = sources[i]
                try:
                    active.append((name, iter(fn())))
                except Exception:
                    continue
            if not active:
                return
            while active:
                next_active: List[Tuple[str, Iterator[Tuple[str, str]]]] = []
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


class SFTIterableDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        pair_iter_factory: Callable[[], Iterator[Tuple[str, str]]],
        max_length: int,
        min_answer_tokens: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pair_iter_factory = pair_iter_factory
        self.max_length = int(max_length)
        self.min_answer_tokens = int(min_answer_tokens)

    def __iter__(self):
        eos = self.tokenizer.eos_token_id
        for prompt, answer in self.pair_iter_factory():
            prompt = normalize_text(prompt)
            answer = normalize_text(answer)
            if not prompt or not answer:
                continue

            prompt_ids = self.tokenizer.encode(prompt + "\n", add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            if len(answer_ids) < self.min_answer_tokens:
                continue

            input_ids = prompt_ids + answer_ids
            if eos is not None:
                input_ids = input_ids + [int(eos)]

            if len(input_ids) > self.max_length:
                # Left-truncate to keep the most recent context while preserving the full answer.
                overflow = len(input_ids) - self.max_length
                if overflow >= len(prompt_ids):
                    continue
                input_ids = input_ids[overflow:]
            # Build labels from the answer span (robust to left-truncation).
            answer_len = len(answer_ids) + (1 if eos is not None else 0)
            answer_start = max(0, len(input_ids) - answer_len)
            labels = [-100] * answer_start + input_ids[answer_start:]
            if not any(v != -100 for v in labels):
                continue

            yield {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        for i, item in enumerate(batch):
            n = len(item["input_ids"])
            input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            attn[i, :n] = 1
            labels[i, :n] = torch.tensor(item["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def _truncate_for_log(text: str, max_chars: int) -> str:
    t = normalize_text(text).replace("\n", " ")
    if len(t) <= max_chars:
        return t
    return t[: max(16, max_chars - 3)] + "..."


DEFAULT_EVAL_PROMPTS: List[str] = [
    "User: Ciao! Puoi salutarmi e dirmi in una frase come mi puoi aiutare?\n\nAssistant:",
    "User: Explain binary search in simple words and state its time complexity.\n\nAssistant:",
    "User: Write a short Python function that checks if a string is a palindrome.\n\nAssistant:",
]


def build_pair_preview_samples(
    sources: List[Tuple[str, Callable[[], Iterator[Tuple[str, str]]]]],
    per_source: int,
) -> List[Tuple[str, str, str]]:
    previews: List[Tuple[str, str, str]] = []
    if not sources or per_source <= 0:
        return previews

    local_first = [s for s in sources if s[0].startswith("local_")] + [s for s in sources if not s[0].startswith("local_")]
    for name, fn in local_first:
        collected = 0
        try:
            for prompt, answer in fn():
                if not prompt or not answer:
                    continue
                previews.append((name, prompt, answer))
                collected += 1
                if collected >= per_source:
                    break
        except Exception:
            continue
    return previews


class PairSampleLoggingCallback(TrainerCallback):
    def __init__(
        self,
        previews: List[Tuple[str, str, str]],
        tokenizer: AutoTokenizer,
        eval_prompts: List[str],
        every_steps: int,
        sample_count: int,
        max_chars: int,
        gen_max_new_tokens: int,
        gen_temperature: float,
        gen_top_p: float,
        seed: int,
    ) -> None:
        self.previews = previews
        self.tokenizer = tokenizer
        self.eval_prompts = [normalize_text(x) for x in eval_prompts if normalize_text(x)]
        self.every_steps = max(1, int(every_steps))
        self.sample_count = max(1, int(sample_count))
        self.max_chars = max(40, int(max_chars))
        self.gen_max_new_tokens = max(8, int(gen_max_new_tokens))
        self.gen_temperature = max(0.0, float(gen_temperature))
        self.gen_top_p = float(max(0.05, min(1.0, float(gen_top_p))))
        self.seed = int(seed)
        self._last_logged_step = -1

    def _pick_eval_prompts(self, step: int) -> List[str]:
        if not self.eval_prompts:
            return []
        rnd = random.Random(self.seed + step + 13_337)
        count = min(self.sample_count, len(self.eval_prompts))
        if len(self.eval_prompts) <= count:
            return list(self.eval_prompts)
        return rnd.sample(self.eval_prompts, count)

    def _make_generation_config(self) -> GenerationConfig:
        do_sample = self.gen_temperature > 0.0
        cfg = GenerationConfig(
            max_new_tokens=int(self.gen_max_new_tokens),
            do_sample=bool(do_sample),
            pad_token_id=int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0),
            eos_token_id=int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None,
        )
        if do_sample:
            cfg.temperature = float(self.gen_temperature)
            cfg.top_p = float(self.gen_top_p)
        return cfg

    def _generate_preview(self, model, prompt: str, step: int) -> str:
        device = next(model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_cfg = self._make_generation_config()

        cpu_state = torch.random.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
            torch.cuda.manual_seed_all(self.seed + step)
        torch.manual_seed(self.seed + step)

        try:
            with torch.no_grad():
                out = model.generate(**inputs, generation_config=gen_cfg)
            generated_ids = out[0][inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return normalize_text(text)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0))
        if hasattr(state, "is_local_process_zero") and not bool(state.is_local_process_zero):
            return control
        if step <= 0 or step == self._last_logged_step:
            return control
        if step % self.every_steps != 0:
            return control
        self._last_logged_step = step
        if not self.previews and not self.eval_prompts:
            return control

        print(f"\n[Sample Preview][lora][step {step}]")

        if self.previews:
            rnd = random.Random(self.seed + step)
            count = min(self.sample_count, len(self.previews))
            picks = rnd.sample(self.previews, count) if len(self.previews) > count else list(self.previews)

            print("[Data]")
            for idx, (src, prompt, answer) in enumerate(picks, start=1):
                p = _truncate_for_log(prompt, self.max_chars)
                a = _truncate_for_log(answer, self.max_chars)
                print(f"{idx}. {src}")
                print(f"   prompt: {p}")
                print(f"   answer: {a}")

        eval_prompts = self._pick_eval_prompts(step)
        model = kwargs.get("model")
        if model is not None and eval_prompts:
            print("[Generation]")
            was_training = bool(model.training)
            had_use_cache = hasattr(getattr(model, "config", None), "use_cache")
            old_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
            model.eval()
            if had_use_cache:
                model.config.use_cache = True
            for idx, prompt in enumerate(eval_prompts, start=1):
                prompt_snippet = _truncate_for_log(prompt, self.max_chars)
                try:
                    completion = self._generate_preview(model, prompt, step)
                    completion_snippet = _truncate_for_log(completion, self.max_chars)
                except Exception as exc:
                    completion_snippet = f"[generation_error] {type(exc).__name__}: {exc}"
                print(f"{idx}. prompt: {prompt_snippet}")
                print(f"   out: {completion_snippet}")
            if had_use_cache:
                model.config.use_cache = old_use_cache
            if was_training:
                model.train()

        print("")
        return control


def detect_target_modules(model) -> List[str]:
    module_names = [name for name, _ in model.named_modules()]
    candidates = [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["c_attn", "c_proj"],
        ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ]
    for cand in candidates:
        found = []
        for c in cand:
            if any(name.endswith(f".{c}") or name == c for name in module_names):
                found.append(c)
        if found:
            return found
    raise RuntimeError(
        "Unable to auto-detect LoRA target modules. Pass --target_modules manually."
    )


def save_interrupt_checkpoint(trainer: Trainer, tokenizer: AutoTokenizer, out_dir: Path, reason: str) -> Path:
    step = int(getattr(trainer.state, "global_step", 0))
    if step > 0 and hasattr(trainer, "_save_checkpoint"):
        try:
            trainer._save_checkpoint(trainer.model, trial=None)  # Keeps optimizer/scheduler state for resume.
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
    ap = argparse.ArgumentParser(description="Train LoRA adapter on instruction/chat datasets")
    ap.add_argument("--model_dir", default="models/lora_base")
    ap.add_argument("--output_dir", default="models/lora_adapter")
    ap.add_argument("--recipe", default="standard", choices=["tiny", "standard", "heavy"])
    ap.add_argument("--hf_source", action="append", default=[], help="Extra source: name|config|split|max_rows")
    ap.add_argument("--disable_hf_data", action="store_true", help="Use only local jsonl globs")
    ap.add_argument("--allow_remote_dataset_code", action="store_true")
    ap.add_argument(
        "--local_jsonl_glob",
        action="append",
        default=["samples/sft/**/*.jsonl", "tiny-llm/samples/sft/**/*.jsonl"],
    )
    ap.add_argument("--disable_local_data", action="store_true")
    ap.add_argument("--max_rows_per_source", type=int, default=0, help="0 = use recipe defaults")
    ap.add_argument("--repeat_sources", action="store_true")

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--min_answer_tokens", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=8_000)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=300)
    ap.add_argument("--save_total_limit", type=int, default=4)
    ap.add_argument("--sample_log_steps", type=int, default=200, help="Print preview samples every N optimizer steps (0 disables)")
    ap.add_argument("--sample_log_count", type=int, default=2, help="How many samples to print each preview event")
    ap.add_argument("--sample_log_max_chars", type=int, default=220, help="Max characters per printed prompt/answer")
    ap.add_argument("--sample_gen_max_new_tokens", type=int, default=128, help="Max generated tokens per eval sample preview")
    ap.add_argument("--sample_gen_temperature", type=float, default=0.0, help="Generation temperature for eval previews (0 = greedy)")
    ap.add_argument("--sample_gen_top_p", type=float, default=0.9, help="Top-p for eval preview generation when temperature > 0")
    ap.add_argument(
        "--sample_eval_prompt",
        action="append",
        default=[],
        help="Extra eval prompt used in sample generation preview (can be repeated)",
    )
    ap.add_argument("--sample_preview_per_source", type=int, default=0, help="How many data preview samples to collect per source at startup (0 disables source prefetch)")
    ap.add_argument("--disable_sample_logging", action="store_true")
    ap.add_argument(
        "--ignore_data_skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, do not iterate/skip old batches (recommended for streaming IterableDataset).",
    )
    ap.add_argument(
        "--use_fused_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fused AdamW when supported by torch/transformers.",
    )
    ap.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 tensor cores on supported NVIDIA GPUs.",
    )
    ap.add_argument(
        "--attn_implementation",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Attention kernel backend. auto=sdpa on CUDA, none on CPU.",
    )
    ap.add_argument(
        "--torch_compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile in Trainer (slower startup).",
    )
    ap.add_argument("--torch_compile_mode", default="max-autotune", help="Compile mode when --torch_compile is enabled.")
    ap.add_argument("--torch_compile_backend", default="auto", help="Compile backend (auto, inductor, aot_eager, eager, ...).")
    ap.add_argument(
        "--throughput_mode",
        action="store_true",
        help="Shorthand for high-throughput runtime settings (compile+tf32+fused optimizer).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--resume_from_checkpoint", default="")

    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="auto", help="Comma-separated list or 'auto'")
    ap.add_argument("--save_merged", action="store_true")
    args = ap.parse_args()

    if args.throughput_mode:
        compile_backend_auto = str(args.torch_compile_backend).strip().lower() in {"", "auto"}
        if platform.system().lower().startswith("win") and compile_backend_auto:
            # Keep throughput mode safe by default on Windows.
            args.torch_compile = False
        else:
            args.torch_compile = True
        args.tf32 = True
        args.use_fused_optimizer = True

    set_seed(int(args.seed))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime(bool(args.tf32))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = int(tokenizer.eos_token_id)
        else:
            print("Warning: tokenizer has no pad_token_id/eos_token_id; defaulting pad_token_id=0")
            tokenizer.pad_token_id = 0

    dtype = resolve_dtype(args.dtype)
    model = load_causal_lm(
        model_id_or_path=args.model_dir,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
    )

    if args.target_modules.strip().lower() == "auto":
        target_modules = detect_target_modules(model)
    else:
        target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
        if not target_modules:
            raise SystemExit("No target modules provided in --target_modules")

    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    sources: List[Tuple[str, Callable[[], Iterator[Tuple[str, str]]]]] = []
    if not args.disable_hf_data:
        for src in RECIPE_SOURCES[args.recipe]:
            max_rows = int(args.max_rows_per_source) if int(args.max_rows_per_source) > 0 else int(src.max_rows)
            resolved = HFSource(src.name, src.config, src.split, max_rows)
            fn = make_hf_pair_iter(resolved, allow_remote_dataset_code=bool(args.allow_remote_dataset_code))
            sources.append((f"hf:{resolved.name}", fn))

        for spec in args.hf_source:
            resolved = parse_hf_source(spec, fallback_max_rows=int(args.max_rows_per_source))
            fn = make_hf_pair_iter(resolved, allow_remote_dataset_code=bool(args.allow_remote_dataset_code))
            sources.append((f"hf:{resolved.name}", fn))

    if not args.disable_local_data:
        for g in args.local_jsonl_glob:
            sources.append((f"local_jsonl:{g}", make_local_jsonl_pair_iter(g)))

    if not sources:
        raise SystemExit("No LoRA training sources found.")

    sample_previews: List[Tuple[str, str, str]] = []
    eval_prompts = list(DEFAULT_EVAL_PROMPTS) + [normalize_text(x) for x in args.sample_eval_prompt if normalize_text(x)]
    if (not args.disable_sample_logging) and int(args.sample_log_steps) > 0:
        preview_per_source = max(0, int(args.sample_preview_per_source))
        if preview_per_source > 0:
            sample_previews = build_pair_preview_samples(
                sources=sources,
                per_source=preview_per_source,
            )
        if sample_previews or eval_prompts:
            print(
                "Sample logging enabled "
                f"(every {int(args.sample_log_steps)} steps, "
                f"{int(args.sample_log_count)} sample(s), "
                f"{len(sample_previews)} data preview rows cached, "
                f"{len(eval_prompts)} eval prompt(s))."
            )
        else:
            print("Sample logging requested but no preview rows could be collected.")

    pair_iter_factory = make_round_robin_pair_iter(
        sources=sources,
        repeat=bool(args.repeat_sources),
        seed=int(args.seed),
    )
    train_dataset = SFTIterableDataset(
        tokenizer=tokenizer,
        pair_iter_factory=pair_iter_factory,
        max_length=int(args.max_length),
        min_answer_tokens=int(args.min_answer_tokens),
    )
    print(
        "SFT dataset: left-truncate to max_length (keep most recent context), "
        "preserve full answers (skip if prompt is too long), "
        "mask prompt tokens in labels."
    )
    collator = SFTCollator(pad_token_id=int(tokenizer.pad_token_id))

    use_bf16 = dtype == torch.bfloat16 and torch.cuda.is_available()
    use_fp16 = dtype == torch.float16 and torch.cuda.is_available()
    optim_name = resolve_optimizer_name(bool(args.use_fused_optimizer))
    compile_mode = str(args.torch_compile_mode).strip() or "max-autotune"
    compile_backend = resolve_torch_compile_backend(str(args.torch_compile_backend))

    print(
        "Runtime config: "
        f"optim={optim_name}, tf32={'on' if bool(args.tf32) else 'off'}, "
        f"torch_compile={'on' if bool(args.torch_compile) else 'off'}"
        f"{f'[{compile_backend}]' if bool(args.torch_compile) and compile_backend else ''}, "
        f"attn={resolve_attn_implementation(str(args.attn_implementation)) or 'default'}"
    )

    targs = make_training_arguments(
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
        optim=optim_name,
        save_safetensors=True,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        ignore_data_skip=bool(args.ignore_data_skip),
        tf32=bool(args.tf32),
        torch_compile=bool(args.torch_compile),
        torch_compile_mode=compile_mode,
        torch_compile_backend=compile_backend,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[
            PairSampleLoggingCallback(
                previews=sample_previews,
                tokenizer=tokenizer,
                eval_prompts=eval_prompts,
                every_steps=int(args.sample_log_steps),
                sample_count=int(args.sample_log_count),
                max_chars=int(args.sample_log_max_chars),
                gen_max_new_tokens=int(args.sample_gen_max_new_tokens),
                gen_temperature=float(args.sample_gen_temperature),
                gen_top_p=float(args.sample_gen_top_p),
                seed=int(args.seed),
            )
        ]
        if (sample_previews or eval_prompts) and (not args.disable_sample_logging) and int(args.sample_log_steps) > 0
        else None,
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
        if args.resume_from_checkpoint:
            print(
                "Resume mode: "
                f"ignore_data_skip={'true' if bool(args.ignore_data_skip) else 'false'}"
            )
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

    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    if args.save_merged:
        if hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            merged_dir = out / "merged_model"
            merged.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))
        else:
            print("Skipping merged save: merge_and_unload not available for this model.")

    run_meta = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "output_dir": str(out.resolve()),
        "recipe": args.recipe,
        "sources": [name for name, _ in sources],
        "max_steps": int(args.max_steps),
        "max_length": int(args.max_length),
        "learning_rate": float(args.learning_rate),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "target_modules": target_modules,
        "dtype": str(dtype),
    }
    (out / "lora_training_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("LoRA training complete.")
    print(f"- output_dir: {out.resolve()}")


if __name__ == "__main__":
    main()
