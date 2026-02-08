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
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
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
    # On current Windows + CUDA stacks, inductor can fail at runtime on some setups.
    if platform.system().lower().startswith("win"):
        return "aot_eager"
    return "inductor"


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


def parse_int_csv(spec: str, fallback: List[int]) -> List[int]:
    vals: List[int] = []
    for part in (spec or "").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = int(p)
            if v > 0:
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return list(fallback)
    return sorted(set(vals))


def probe_shape_fits(model, vocab_size: int, batch_size: int, seq_len: int) -> Tuple[bool, int]:
    if not torch.cuda.is_available():
        return True, 0

    device = torch.device("cuda")
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        input_ids = torch.randint(
            low=0,
            high=max(10, int(vocab_size)),
            size=(int(batch_size), int(seq_len)),
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        peak = int(torch.cuda.max_memory_allocated(device))
        del out, loss, input_ids, attention_mask, labels
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return True, peak
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda error" in msg:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False, 0
        raise


def auto_tune_training_shape(
    model,
    tokenizer,
    init_batch_size: int,
    init_block_size: int,
    batch_candidates: List[int],
    block_candidates: List[int],
    target_vram_frac: float,
    max_trials: int,
) -> Tuple[int, int]:
    if not torch.cuda.is_available():
        return int(init_batch_size), int(init_block_size)

    model.to("cuda")
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or getattr(model.config, "vocab_size", 50257))
    max_positions = int(getattr(model.config, "max_position_embeddings", 32768) or 32768)
    blocks = [b for b in block_candidates if 16 <= int(b) <= max_positions]
    if not blocks:
        blocks = [int(init_block_size)]

    batches = [b for b in batch_candidates if int(b) > 0]
    if not batches:
        batches = [int(init_batch_size)]

    combos = sorted(
        {(int(b), int(s)) for b in batches for s in blocks},
        key=lambda x: (x[0] * x[1], x[1], x[0]),
        reverse=True,
    )
    if int(max_trials) > 0:
        combos = combos[: int(max_trials)]

    total_mem = int(torch.cuda.get_device_properties(0).total_memory)
    fits: List[Tuple[int, int, int, float]] = []

    print(
        "Auto-tuning GPU shape: "
        f"{len(combos)} trial(s), target VRAM <= {max(0.5, min(0.99, float(target_vram_frac))):.2f}"
    )
    for bs, seq in combos:
        ok, peak = probe_shape_fits(model, vocab_size=vocab_size, batch_size=bs, seq_len=seq)
        if not ok:
            print(f"- bs={bs}, block={seq}: OOM")
            continue
        frac = float(peak) / float(total_mem) if total_mem > 0 else 0.0
        fits.append((bs, seq, peak, frac))
        print(f"- bs={bs}, block={seq}: OK (peak={peak // (1024**2)} MB, frac={frac:.3f})")

    if not fits:
        print("Auto-tuning found no valid shape; keeping current batch/block values.")
        return int(init_batch_size), int(init_block_size)

    target = max(0.5, min(0.99, float(target_vram_frac)))
    preferred = [x for x in fits if x[3] <= target]
    chosen_pool = preferred if preferred else fits
    chosen = max(chosen_pool, key=lambda x: (x[0] * x[1], x[1], x[0]))
    chosen_bs, chosen_block, chosen_peak, chosen_frac = chosen

    print(
        "Auto-tuning selected: "
        f"batch_size={chosen_bs}, block_size={chosen_block} "
        f"(peak={chosen_peak // (1024**2)} MB, frac={chosen_frac:.3f})"
    )
    return int(chosen_bs), int(chosen_block)


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


def _truncate_for_log(text: str, max_chars: int) -> str:
    t = normalize_text(text).replace("\n", " ")
    if len(t) <= max_chars:
        return t
    return t[: max(16, max_chars - 3)] + "..."


DEFAULT_EVAL_PROMPTS: List[str] = [
    "Write a short friendly greeting in Italian for a new user.",
    "Explain in simple words what binary search is and when to use it.",
    "Give 3 practical tips to improve Python code quality.",
]


def build_text_preview_samples(
    sources: List[Tuple[str, Callable[[], Iterator[str]]]],
    per_source: int,
) -> List[Tuple[str, str]]:
    previews: List[Tuple[str, str]] = []
    if not sources or per_source <= 0:
        return previews

    local_first = [s for s in sources if s[0].startswith("local_")] + [s for s in sources if not s[0].startswith("local_")]
    for name, fn in local_first:
        collected = 0
        try:
            for text in fn():
                if not text:
                    continue
                previews.append((name, text))
                collected += 1
                if collected >= per_source:
                    break
        except Exception:
            continue
    return previews


class TextSampleLoggingCallback(TrainerCallback):
    def __init__(
        self,
        previews: List[Tuple[str, str]],
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
        self.gen_top_p = float(max(0.05, min(1.0, gen_top_p)))
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

    def _generate_preview(self, model, prompt: str, step: int) -> str:
        device = next(model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        do_sample = self.gen_temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": int(self.gen_max_new_tokens),
            "pad_token_id": int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0),
            "eos_token_id": int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None,
            "do_sample": bool(do_sample),
        }
        if do_sample:
            gen_kwargs["temperature"] = float(self.gen_temperature)
            gen_kwargs["top_p"] = float(self.gen_top_p)

        cpu_state = torch.random.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
            torch.cuda.manual_seed_all(self.seed + step)
        torch.manual_seed(self.seed + step)

        try:
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
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

        print(f"\n[Sample Preview][base][step {step}]")

        if self.previews:
            rnd = random.Random(self.seed + step)
            count = min(self.sample_count, len(self.previews))
            picks = rnd.sample(self.previews, count) if len(self.previews) > count else list(self.previews)
            print("[Data]")
            for idx, (src, text) in enumerate(picks, start=1):
                snippet = _truncate_for_log(text, self.max_chars)
                print(f"{idx}. {src}: {snippet}")

        eval_prompts = self._pick_eval_prompts(step)
        model = kwargs.get("model")
        if model is not None and eval_prompts:
            print("[Generation]")
            was_training = bool(model.training)
            had_use_cache = hasattr(model.config, "use_cache")
            old_use_cache = getattr(model.config, "use_cache", None)
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

    ap.add_argument("--block_size", type=int, default=1280)
    ap.add_argument("--auto_tune_shape", action="store_true", help="Auto-find the largest safe batch/block shape on GPU")
    ap.add_argument("--auto_tune_target_vram_frac", type=float, default=0.93, help="Max VRAM fraction target for auto-tuned shape")
    ap.add_argument("--auto_tune_batch_candidates", default="2,3,4,5,6", help="CSV list of batch sizes to probe")
    ap.add_argument("--auto_tune_block_candidates", default="1024,1280,1536,1792,2048", help="CSV list of block sizes to probe")
    ap.add_argument("--auto_tune_max_trials", type=int, default=24, help="Max number of shape probes")
    ap.add_argument("--max_steps", type=int, default=30_000)
    ap.add_argument("--per_device_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=4)
    ap.add_argument("--sample_log_steps", type=int, default=200, help="Print preview samples every N optimizer steps (0 disables)")
    ap.add_argument("--sample_log_count", type=int, default=2, help="How many samples to print each preview event")
    ap.add_argument("--sample_log_max_chars", type=int, default=220, help="Max characters per printed sample")
    ap.add_argument("--sample_gen_max_new_tokens", type=int, default=96, help="Max generated tokens per eval sample preview")
    ap.add_argument("--sample_gen_temperature", type=float, default=0.7, help="Generation temperature for eval previews (0 = greedy)")
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
        help="Enable torch.compile in Trainer (good for long runs, slower startup).",
    )
    ap.add_argument(
        "--torch_compile_mode",
        default="max-autotune",
        help="Compile mode when --torch_compile is enabled (e.g. max-autotune, reduce-overhead).",
    )
    ap.add_argument(
        "--torch_compile_backend",
        default="auto",
        help="Compile backend (auto, inductor, aot_eager, eager, ...).",
    )
    ap.add_argument(
        "--throughput_mode",
        action="store_true",
        help="Shorthand for high-throughput runtime settings (enables torch_compile, tf32 and fused optimizer).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--resume_from_checkpoint", default="")
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

    dtype = resolve_dtype(args.dtype)
    model = load_causal_lm(
        model_id_or_path=args.model_dir,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
    )
    model.config.use_cache = False

    if tokenizer.vocab_size > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(tokenizer.vocab_size)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.auto_tune_shape:
        tuned_bs, tuned_block = auto_tune_training_shape(
            model=model,
            tokenizer=tokenizer,
            init_batch_size=int(args.per_device_batch_size),
            init_block_size=int(args.block_size),
            batch_candidates=parse_int_csv(args.auto_tune_batch_candidates, fallback=[int(args.per_device_batch_size)]),
            block_candidates=parse_int_csv(args.auto_tune_block_candidates, fallback=[int(args.block_size)]),
            target_vram_frac=float(args.auto_tune_target_vram_frac),
            max_trials=int(args.auto_tune_max_trials),
        )
        args.per_device_batch_size = int(tuned_bs)
        args.block_size = int(tuned_block)
        print(
            "Using auto-tuned shape: "
            f"per_device_batch_size={int(args.per_device_batch_size)}, "
            f"block_size={int(args.block_size)}"
        )

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

    sample_previews: List[Tuple[str, str]] = []
    eval_prompts = list(DEFAULT_EVAL_PROMPTS) + [normalize_text(x) for x in args.sample_eval_prompt if normalize_text(x)]
    if (not args.disable_sample_logging) and int(args.sample_log_steps) > 0:
        preview_per_source = max(0, int(args.sample_preview_per_source))
        if preview_per_source > 0:
            sample_previews = build_text_preview_samples(
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
        train_dataset=dataset,
        data_collator=default_data_collator,
        callbacks=[
            TextSampleLoggingCallback(
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
