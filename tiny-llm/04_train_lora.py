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
from sft_data_quality import (
    JsonlFileReport,
    JsonlLineError,
    SftHygieneConfig,
    apply_sft_hygiene,
    format_validation_table,
    normalize_text_preserve_whitespace,
    parse_jsonl_object_line,
)
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

DEFAULT_LOCAL_JSONL_GLOBS: List[str] = [
    "samples/sft/**/*.jsonl",
    "tiny-llm/samples/sft/**/*.jsonl",
]


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


def resolve_4bit_compute_dtype(name: str, fallback: torch.dtype) -> torch.dtype:
    n = (name or "auto").strip().lower()
    if n == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return fallback
    if n in {"float16", "fp16"}:
        return torch.float16
    if n in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if n in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported 4-bit compute dtype: {name}")


def prepare_model_for_optional_4bit_training(model, use_gradient_checkpointing: bool):
    try:
        from peft import prepare_model_for_kbit_training
    except Exception as exc:
        raise RuntimeError(
            "4-bit quantization requested but this PEFT version has no prepare_model_for_kbit_training. "
            "Upgrade peft or disable --use_4bit."
        ) from exc
    try:
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
        )
    except TypeError:
        return prepare_model_for_kbit_training(model)


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
    use_4bit: bool,
    bnb_4bit_quant_type: str,
    bnb_4bit_compute_dtype: torch.dtype,
    bnb_4bit_use_double_quant: bool,
):
    kwargs = {"trust_remote_code": bool(trust_remote_code)}
    attn_impl = resolve_attn_implementation(attn_implementation)
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    load_dtype = dtype
    if bool(use_4bit):
        if not torch.cuda.is_available():
            raise RuntimeError("--use_4bit requires CUDA.")
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "4-bit quantization requested but BitsAndBytesConfig is unavailable. "
                "Upgrade transformers and install bitsandbytes."
            ) from exc
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "4-bit quantization requested but bitsandbytes is not installed. "
                "Install bitsandbytes or disable --use_4bit."
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(bnb_4bit_quant_type),
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bool(bnb_4bit_use_double_quant),
        )
        kwargs["device_map"] = "auto"
        kwargs["low_cpu_mem_usage"] = True
        load_dtype = bnb_4bit_compute_dtype
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            dtype=load_dtype,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=load_dtype,
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


def has_tokenizer_chat_template(tokenizer) -> bool:
    tpl = getattr(tokenizer, "chat_template", None)
    return bool(hasattr(tokenizer, "apply_chat_template") and tpl)


def resolve_chat_format_mode(requested_mode: str, tokenizer) -> str:
    mode = (requested_mode or "legacy").strip().lower()
    if mode not in {"legacy", "tokenizer", "auto"}:
        raise ValueError(f"Unsupported --chat_format value: {requested_mode}")
    if mode == "auto":
        return "tokenizer" if has_tokenizer_chat_template(tokenizer) else "legacy"
    if mode == "tokenizer" and (not has_tokenizer_chat_template(tokenizer)):
        raise ValueError("Tokenizer chat template not available; cannot use --chat_format tokenizer")
    return mode


def _legacy_prompt_builder(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        content = normalize_text(str(msg.get("content", "")))
        if not content:
            continue
        prefix = "System" if role == "system" else ("Assistant" if role == "assistant" else "User")
        lines.append(f"{prefix}: {content}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def make_prompt_builder(chat_mode: str, tokenizer) -> Callable[[List[Dict[str, str]]], str]:
    mode = resolve_chat_format_mode(chat_mode, tokenizer)
    if mode == "legacy":
        return _legacy_prompt_builder

    def _tokenizer_prompt_builder(messages: List[Dict[str, str]]) -> str:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, list):
            rendered = "".join(str(x) for x in rendered)
        return normalize_text_preserve_whitespace(str(rendered))

    return _tokenizer_prompt_builder


def pairs_from_messages(
    messages: object,
    prompt_builder: Callable[[List[Dict[str, str]]], str],
) -> List[Tuple[str, str]]:
    if not isinstance(messages, list):
        return []
    cleaned: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        raw_content = str(m.get("content", ""))
        if role == "assistant":
            content = normalize_text_preserve_whitespace(raw_content)
        else:
            content = normalize_text(raw_content)
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    if not cleaned:
        return []

    pairs: List[Tuple[str, str]] = []
    history: List[Dict[str, str]] = []
    for msg in cleaned:
        if msg["role"] == "assistant":
            prompt = prompt_builder(history)
            answer = normalize_text_preserve_whitespace(msg["content"])
            if prompt.strip() and answer:
                pairs.append((prompt, answer))
        history.append(msg)
    return pairs


def row_to_pairs(
    row: Dict[str, object],
    prompt_builder: Callable[[List[Dict[str, str]]], str],
) -> List[Tuple[str, str]]:
    pairs = pairs_from_messages(row.get("messages"), prompt_builder=prompt_builder)
    if pairs:
        return pairs

    if isinstance(row.get("question"), str) and isinstance(row.get("response"), str):
        messages: List[Dict[str, str]] = []
        sys_prompt = normalize_text(str(row.get("system_prompt", "")))
        question = normalize_text(str(row["question"]))
        answer = normalize_text_preserve_whitespace(str(row["response"]))
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": question})
        prompt = prompt_builder(messages)
        if prompt and answer:
            return [(prompt, answer)]
        return []

    if isinstance(row.get("instruction"), str):
        instruction = normalize_text(str(row.get("instruction", "")))
        inp = normalize_text(str(row.get("input", "")))
        ctx = normalize_text(str(row.get("context", "")))
        answer = normalize_text_preserve_whitespace(str(row.get("output", row.get("response", ""))))
        if answer:
            user_text = instruction
            if inp:
                user_text = f"{user_text}\n\nInput: {inp}"
            if ctx:
                user_text = f"{user_text}\n\nContext: {ctx}"
            prompt = prompt_builder([{"role": "user", "content": user_text}])
            if prompt:
                return [(prompt, answer)]

    if isinstance(row.get("prompt"), str) and isinstance(row.get("response"), str):
        prompt = normalize_text(str(row["prompt"]))
        answer = normalize_text_preserve_whitespace(str(row["response"]))
        rendered_prompt = prompt_builder([{"role": "user", "content": prompt}])
        if rendered_prompt and answer:
            return [(rendered_prompt, answer)]

    return []


def make_hf_pair_iter(
    source: HFSource,
    allow_remote_dataset_code: bool,
    prompt_builder: Callable[[List[Dict[str, str]]], str],
    hygiene_cfg: SftHygieneConfig,
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
            if not isinstance(row, dict):
                continue
            for prompt, answer in row_to_pairs(row, prompt_builder=prompt_builder):
                h = apply_sft_hygiene(row, answer, hygiene_cfg)
                if prompt and h.keep and h.answer:
                    yield (prompt, h.answer)
            if source.max_rows > 0 and rows >= source.max_rows:
                break

    return _iter


def make_local_jsonl_pair_iter(
    glob_pattern: str,
    prompt_builder: Callable[[List[Dict[str, str]]], str],
    hygiene_cfg: SftHygieneConfig,
    strict_jsonl: bool,
) -> Callable[[], Iterator[Tuple[str, str]]]:
    def _iter() -> Iterator[Tuple[str, str]]:
        for p in sorted(Path(".").glob(glob_pattern)):
            if not p.is_file():
                continue
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for line_no, raw_line in enumerate(f, start=1):
                        row, err = parse_jsonl_object_line(raw_line)
                        if err:
                            if strict_jsonl:
                                raise ValueError(f"{p}:{line_no}: {err}")
                            continue
                        if row is None:
                            continue
                        for prompt, answer in row_to_pairs(row, prompt_builder=prompt_builder):
                            h = apply_sft_hygiene(row, answer, hygiene_cfg)
                            if prompt and h.keep and h.answer:
                                yield (prompt, h.answer)
            except Exception:
                if strict_jsonl:
                    raise
                continue

    return _iter


def pair_signature(prompt: str, answer: str) -> Tuple[str, str]:
    # Normalize whitespace-only differences so exact duplicate SFT pairs are detectable.
    p = re.sub(r"\s+", " ", normalize_text_preserve_whitespace(prompt)).strip()
    a = re.sub(r"\s+", " ", normalize_text_preserve_whitespace(answer)).strip()
    return (p, a)


def build_jsonl_validation_report(
    path: Path,
    prompt_builder: Callable[[List[Dict[str, str]]], str],
    hygiene_cfg: SftHygieneConfig,
) -> JsonlFileReport:
    report = JsonlFileReport(path=path)
    seen_pairs: set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            report.total_lines += 1
            row, err = parse_jsonl_object_line(raw_line)
            if err:
                report.invalid_lines += 1
                report.errors.append(JsonlLineError(line_no=line_no, message=err))
                continue

            report.valid_lines += 1
            if row is None:
                continue

            for prompt, answer in row_to_pairs(row, prompt_builder=prompt_builder):
                if not prompt or not answer:
                    continue
                h = apply_sft_hygiene(row, answer, hygiene_cfg)
                if h.keep and h.answer:
                    sig = pair_signature(prompt, h.answer)
                    if sig in seen_pairs:
                        report.duplicate_examples += 1
                    else:
                        seen_pairs.add(sig)
                    report.loaded_examples += 1
                else:
                    report.filtered_examples += 1
    return report


def validate_local_jsonl_sources(
    globs: List[str],
    prompt_builder: Callable[[List[Dict[str, str]]], str],
    hygiene_cfg: SftHygieneConfig,
) -> Tuple[List[JsonlFileReport], Dict[str, int]]:
    reports: List[JsonlFileReport] = []
    loaded_per_source: Dict[str, int] = {}
    for g in globs:
        src_name = f"local_jsonl:{g}"
        loaded_per_source[src_name] = 0
        files = [p for p in sorted(Path(".").glob(g)) if p.is_file()]
        for p in files:
            rep = build_jsonl_validation_report(
                path=p,
                prompt_builder=prompt_builder,
                hygiene_cfg=hygiene_cfg,
            )
            reports.append(rep)
            loaded_per_source[src_name] += int(rep.loaded_examples)
    return reports, loaded_per_source


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
            answer = normalize_text_preserve_whitespace(answer)
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
        default=[],
        help="Local JSONL glob(s). If omitted, defaults to samples/sft globs.",
    )
    ap.add_argument("--disable_local_data", action="store_true")
    ap.add_argument("--max_rows_per_source", type=int, default=0, help="0 = use recipe defaults")
    ap.add_argument("--repeat_sources", action="store_true")
    ap.add_argument(
        "--validate_data",
        action="store_true",
        help="Strictly validate local JSONL before training and fail on malformed lines.",
    )
    ap.add_argument(
        "--min_loaded_examples",
        type=int,
        default=200,
        help="Minimum loaded examples required when --validate_data is enabled.",
    )
    ap.add_argument(
        "--allow_small_dataset",
        action="store_true",
        help="Allow training with fewer than --min_loaded_examples after filtering.",
    )
    ap.add_argument(
        "--fail_on_duplicate_examples",
        action="store_true",
        help="Abort when duplicate loaded examples ratio exceeds --max_duplicate_example_ratio.",
    )
    ap.add_argument(
        "--max_duplicate_example_ratio",
        type=float,
        default=0.20,
        help="Max allowed duplicate ratio (duplicate_examples / loaded_examples) when --fail_on_duplicate_examples is set.",
    )
    ap.add_argument(
        "--chat_format",
        default="legacy",
        choices=["legacy", "tokenizer", "auto"],
        help="Prompt format used to build SFT prompts from role messages.",
    )
    ap.add_argument(
        "--code_fence_hygiene",
        default="off",
        choices=["off", "reject", "normalize"],
        help="Code-fence hygiene for Python-like assistant outputs in SFT data.",
    )
    ap.add_argument(
        "--code_fence_language",
        default="python",
        help="Language tag used when --code_fence_hygiene normalize wraps code blocks.",
    )
    ap.add_argument(
        "--reject_no_markdown_code_examples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject code-like examples whose system/user prompt asks for no markdown.",
    )

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
    ap.add_argument(
        "--use_4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the base model in 4-bit (QLoRA). Requires CUDA + bitsandbytes.",
    )
    ap.add_argument(
        "--bnb_4bit_quant_type",
        default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit quantization type when --use_4bit is enabled.",
    )
    ap.add_argument(
        "--bnb_4bit_compute_dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Compute dtype for 4-bit training when --use_4bit is enabled.",
    )
    ap.add_argument(
        "--bnb_4bit_use_double_quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable double quantization for 4-bit training.",
    )
    ap.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 auto: 2 on CUDA, else 0). Increase if GPU is underutilized.",
    )
    ap.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor (used only when workers > 0).",
    )
    ap.add_argument(
        "--dataloader_persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between steps when workers > 0.",
    )
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
    local_globs = list(args.local_jsonl_glob) if args.local_jsonl_glob else list(DEFAULT_LOCAL_JSONL_GLOBS)
    hygiene_cfg = SftHygieneConfig(
        code_fence_mode=str(args.code_fence_hygiene),
        fence_language=str(args.code_fence_language),
        reject_no_markdown_code_instructions=bool(args.reject_no_markdown_code_examples),
    )

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
    prompt_builder = make_prompt_builder(chat_mode=str(args.chat_format), tokenizer=tokenizer)
    effective_chat_format = resolve_chat_format_mode(str(args.chat_format), tokenizer=tokenizer)
    print(f"Chat format: {effective_chat_format}")
    if effective_chat_format == "legacy" and has_tokenizer_chat_template(tokenizer):
        print(
            "Warning: tokenizer chat template is available but disabled; "
            "this can cause train/inference format mismatch."
        )

    if args.local_jsonl_glob:
        print(
            f"Local JSONL globs (append mode, {len(args.local_jsonl_glob)} values): "
            + ", ".join(args.local_jsonl_glob)
        )

    validated_reports: List[JsonlFileReport] = []
    loaded_per_source: Dict[str, int] = {}
    if bool(args.validate_data) and (not bool(args.disable_local_data)):
        validated_reports, loaded_per_source = validate_local_jsonl_sources(
            globs=local_globs,
            prompt_builder=prompt_builder,
            hygiene_cfg=hygiene_cfg,
        )
        if validated_reports:
            print("")
            print("Strict JSONL validation summary")
            print(format_validation_table(validated_reports))
        else:
            print("Strict JSONL validation: no local files matched configured globs.")

        all_errors: List[Tuple[Path, JsonlLineError]] = []
        for rep in validated_reports:
            for err in rep.errors:
                all_errors.append((rep.path, err))
        if all_errors:
            print("")
            print("Validation errors:")
            for p, err in all_errors:
                print(f"- {p}:{err.line_no}: {err.message}")
            raise SystemExit("Aborting: strict JSONL validation failed.")

        if loaded_per_source:
            print("")
            print("Loaded examples per local source (after hygiene):")
            for src_name in sorted(loaded_per_source.keys()):
                print(f"- {src_name}: {loaded_per_source[src_name]}")

        total_loaded = sum(int(x.loaded_examples) for x in validated_reports)
        total_duplicates = sum(int(x.duplicate_examples) for x in validated_reports)
        duplicate_ratio = (float(total_duplicates) / float(total_loaded)) if total_loaded > 0 else 0.0
        print(
            "Duplicate loaded examples: "
            f"{total_duplicates}/{total_loaded} ({duplicate_ratio:.2%})"
        )
        if bool(args.fail_on_duplicate_examples) and duplicate_ratio > float(args.max_duplicate_example_ratio):
            raise SystemExit(
                f"Aborting: duplicate loaded examples ratio {duplicate_ratio:.2%} exceeds "
                f"--max_duplicate_example_ratio={float(args.max_duplicate_example_ratio):.2%}."
            )
        if total_duplicates > 0 and duplicate_ratio > float(args.max_duplicate_example_ratio):
            print(
                "Warning: duplicate loaded examples ratio "
                f"{duplicate_ratio:.2%} exceeds --max_duplicate_example_ratio="
                f"{float(args.max_duplicate_example_ratio):.2%}."
            )

        if (total_loaded < int(args.min_loaded_examples)) and (not bool(args.allow_small_dataset)):
            raise SystemExit(
                f"Aborting: loaded examples after filtering = {total_loaded}, "
                f"below --min_loaded_examples={int(args.min_loaded_examples)}. "
                "Use --allow_small_dataset to override."
            )
        if total_loaded < int(args.min_loaded_examples):
            print(
                f"Warning: loaded examples after filtering ({total_loaded}) are below "
                f"--min_loaded_examples={int(args.min_loaded_examples)} (override accepted)."
            )
    elif bool(args.validate_data):
        print("Strict JSONL validation skipped: local data is disabled.")

    dtype = resolve_dtype(args.dtype)
    bnb_compute_dtype = resolve_4bit_compute_dtype(
        str(args.bnb_4bit_compute_dtype),
        fallback=dtype,
    )
    if bool(args.use_4bit):
        print(
            "QLoRA 4-bit enabled: "
            f"quant={str(args.bnb_4bit_quant_type)}, "
            f"compute_dtype={str(bnb_compute_dtype)}, "
            f"double_quant={'on' if bool(args.bnb_4bit_use_double_quant) else 'off'}"
        )
    model = load_causal_lm(
        model_id_or_path=args.model_dir,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
        use_4bit=bool(args.use_4bit),
        bnb_4bit_quant_type=str(args.bnb_4bit_quant_type),
        bnb_4bit_compute_dtype=bnb_compute_dtype,
        bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
    )
    if bool(args.use_4bit):
        model = prepare_model_for_optional_4bit_training(
            model,
            use_gradient_checkpointing=bool(args.gradient_checkpointing),
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
            fn = make_hf_pair_iter(
                resolved,
                allow_remote_dataset_code=bool(args.allow_remote_dataset_code),
                prompt_builder=prompt_builder,
                hygiene_cfg=hygiene_cfg,
            )
            sources.append((f"hf:{resolved.name}", fn))

        for spec in args.hf_source:
            resolved = parse_hf_source(spec, fallback_max_rows=int(args.max_rows_per_source))
            fn = make_hf_pair_iter(
                resolved,
                allow_remote_dataset_code=bool(args.allow_remote_dataset_code),
                prompt_builder=prompt_builder,
                hygiene_cfg=hygiene_cfg,
            )
            sources.append((f"hf:{resolved.name}", fn))

    if not args.disable_local_data:
        for g in local_globs:
            sources.append(
                (
                    f"local_jsonl:{g}",
                    make_local_jsonl_pair_iter(
                        g,
                        prompt_builder=prompt_builder,
                        hygiene_cfg=hygiene_cfg,
                        strict_jsonl=bool(args.validate_data),
                    ),
                )
            )

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

    compute_dtype = bnb_compute_dtype if bool(args.use_4bit) else dtype
    use_bf16 = compute_dtype == torch.bfloat16 and torch.cuda.is_available()
    use_fp16 = compute_dtype == torch.float16 and torch.cuda.is_available()
    optim_name = resolve_optimizer_name(bool(args.use_fused_optimizer))
    compile_mode = str(args.torch_compile_mode).strip() or "max-autotune"
    compile_backend = resolve_torch_compile_backend(str(args.torch_compile_backend))
    if int(args.dataloader_num_workers) < 0:
        dataloader_num_workers = 2 if torch.cuda.is_available() else 0
    else:
        dataloader_num_workers = int(args.dataloader_num_workers)
    # NOTE: this training pipeline builds iterable factories with local callables.
    # On Windows, DataLoader multiprocessing requires picklable worker state and
    # fails with "Can't pickle local object ...". Force single-process loading.
    if platform.system().lower().startswith("win") and dataloader_num_workers > 0:
        print(
            "DataLoader workers > 0 are not supported on Windows with this IterableDataset "
            "(non-picklable local iter factory). Falling back to dataloader_num_workers=0."
        )
        dataloader_num_workers = 0
    dataloader_persistent_workers = bool(args.dataloader_persistent_workers) and dataloader_num_workers > 0
    dataloader_prefetch_factor = int(args.dataloader_prefetch_factor) if dataloader_num_workers > 0 else None

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
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_persistent_workers=dataloader_persistent_workers,
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
        "compute_dtype": str(compute_dtype),
        "use_4bit": bool(args.use_4bit),
        "bnb_4bit_quant_type": str(args.bnb_4bit_quant_type),
        "bnb_4bit_compute_dtype": str(bnb_compute_dtype),
        "bnb_4bit_use_double_quant": bool(args.bnb_4bit_use_double_quant),
        "chat_format": str(effective_chat_format),
        "validate_data": bool(args.validate_data),
        "fail_on_duplicate_examples": bool(args.fail_on_duplicate_examples),
        "max_duplicate_example_ratio": float(args.max_duplicate_example_ratio),
        "code_fence_hygiene": str(args.code_fence_hygiene),
        "code_fence_language": str(args.code_fence_language),
        "reject_no_markdown_code_examples": bool(args.reject_no_markdown_code_examples),
        "dataloader_num_workers": int(dataloader_num_workers),
        "dataloader_prefetch_factor": int(args.dataloader_prefetch_factor),
        "dataloader_persistent_workers": bool(dataloader_persistent_workers),
    }
    (out / "lora_training_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("LoRA training complete.")
    print(f"- output_dir: {out.resolve()}")


if __name__ == "__main__":
    main()
