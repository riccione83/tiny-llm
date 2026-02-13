#!/usr/bin/env python3
"""
Tiny formatting/correctness regression suite.

Default backend is `mock` for fast no-GPU checks. Use `--backend hf` to run on
an actual model (base model or base+LoRA adapter).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


@dataclass
class RegressionCase:
    name: str
    messages: List[Dict[str, str]]
    check: Callable[[str], Tuple[bool, str]]


def _has_python_fence(text: str) -> bool:
    t = (text or "")
    return ("```python" in t.lower()) and (t.count("```") >= 2)


def _check_python_code_fence(text: str) -> Tuple[bool, str]:
    t = (text or "").strip()
    if not _has_python_fence(t):
        return False, "missing ```python fenced block"
    if "def " not in t:
        return False, "missing function definition"
    return True, ""


def _check_bullets(text: str) -> Tuple[bool, str]:
    lines = [x for x in (text or "").splitlines() if x.strip()]
    bullets = sum(1 for ln in lines if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", ln))
    if bullets < 3:
        return False, f"expected >=3 bullet lines, got {bullets}"
    return True, ""


def _check_one_sentence(text: str) -> Tuple[bool, str]:
    t = (text or "").strip()
    if not t:
        return False, "empty output"
    sentences = [x for x in re.split(r"[.!?]+", t) if x.strip()]
    if len(sentences) != 1:
        return False, f"expected exactly 1 sentence, got {len(sentences)}"
    return True, ""


def _check_basic_math(text: str) -> Tuple[bool, str]:
    t = (text or "").strip()
    if re.fullmatch(r"\s*133\s*", t):
        return True, ""
    if "133" in t:
        return True, ""
    return False, "expected result 133"


def default_cases() -> List[RegressionCase]:
    return [
        RegressionCase(
            name="python_fence",
            messages=[
                {
                    "role": "user",
                    "content": "Write a Python function factorial(n). Return only a fenced python code block.",
                }
            ],
            check=_check_python_code_fence,
        ),
        RegressionCase(
            name="bullets",
            messages=[
                {"role": "user", "content": "Give 3 concise bullet points on why unit tests matter."}
            ],
            check=_check_bullets,
        ),
        RegressionCase(
            name="one_sentence",
            messages=[
                {"role": "user", "content": "In one sentence, what is version control?"}
            ],
            check=_check_one_sentence,
        ),
        RegressionCase(
            name="math",
            messages=[
                {"role": "user", "content": "What is 19 * 7? Reply with the answer."}
            ],
            check=_check_basic_math,
        ),
    ]


class MockGenerator:
    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        prompt = (messages[-1].get("content", "") if messages else "").lower()
        if "factorial" in prompt:
            return "```python\ndef factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)\n```"
        if "bullet" in prompt:
            return "- Catches regressions early.\n- Improves refactor safety.\n- Documents expected behavior."
        if "one sentence" in prompt:
            return "Version control tracks changes and enables safe collaboration."
        if "19 * 7" in prompt or "19*7" in prompt:
            return "133"
        return "I don't know."


class HFGenerator:
    def __init__(
        self,
        model_dir: str,
        adapter_dir: str,
        device: str,
        max_new_tokens: int,
        chat_format: str,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if self._tok.pad_token_id is None and self._tok.eos_token_id is not None:
            self._tok.pad_token_id = int(self._tok.eos_token_id)

        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        self._device = resolved_device

        dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
        if adapter_dir:
            model = PeftModel.from_pretrained(model, adapter_dir)
        model.to(resolved_device)
        model.eval()
        self._model = model
        self._max_new_tokens = max(16, int(max_new_tokens))
        mode = (chat_format or "auto").strip().lower()
        if mode == "auto":
            mode = "tokenizer" if bool(getattr(self._tok, "chat_template", None)) else "legacy"
        self._chat_format = mode

    def _build_prompt(self, messages: Sequence[Dict[str, str]]) -> str:
        if self._chat_format == "tokenizer" and hasattr(self._tok, "apply_chat_template"):
            prompt = self._tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
            return prompt if isinstance(prompt, str) else "".join(str(x) for x in prompt)
        lines = []
        for m in messages:
            role = str(m.get("role", "user")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role == "system":
                p = "System"
            elif role == "assistant":
                p = "Assistant"
            else:
                p = "User"
            lines.append(f"{p}: {content}")
        lines.append("Assistant:")
        return "\n\n".join(lines)

    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        prompt = self._build_prompt(messages)
        inputs = self._tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                eos_token_id=self._tok.eos_token_id,
                pad_token_id=self._tok.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


def run_suite(generator, cases: Sequence[RegressionCase]) -> Tuple[int, int, List[Tuple[str, str]]]:
    failures: List[Tuple[str, str]] = []
    passed = 0
    for case in cases:
        out = generator.generate(case.messages)
        ok, reason = case.check(out)
        if ok:
            passed += 1
            print(f"[OK]   {case.name}")
        else:
            failures.append((case.name, reason))
            print(f"[FAIL] {case.name}: {reason}")
        print(f"  out: {(out or '').strip()[:240]}")
    return passed, len(cases), failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiny formatting/correctness regression suite")
    ap.add_argument("--backend", default="mock", choices=["mock", "hf"])
    ap.add_argument("--model_dir", default="models/base_trained")
    ap.add_argument("--adapter_dir", default="")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--chat_format", default="auto", choices=["auto", "legacy", "tokenizer"])
    args = ap.parse_args()

    if args.backend == "hf":
        gen = HFGenerator(
            model_dir=str(args.model_dir),
            adapter_dir=str(args.adapter_dir),
            device=str(args.device),
            max_new_tokens=int(args.max_new_tokens),
            chat_format=str(args.chat_format),
        )
    else:
        gen = MockGenerator()

    cases = default_cases()
    passed, total, failures = run_suite(gen, cases)
    print(f"PASS {passed}/{total}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
