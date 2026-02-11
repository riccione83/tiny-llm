from typing import Dict, List

import os
from pathlib import Path
import json
import torch

from .tiny_backend import TinyLocalLLM


class HFLocalLLM:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        use_chat_template: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.use_chat_template = bool(use_chat_template)
        self._tok = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _from_pretrained_with_dtype(model_name: str, dtype: torch.dtype, device_map):
        from transformers import AutoModelForCausalLM  # type: ignore

        kwargs = {"device_map": device_map}
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)

    def _lazy_load(self) -> None:
        if self._tok is not None and self._model is not None:
            return
        from transformers import AutoTokenizer  # type: ignore
        from transformers.utils import logging as hf_logging  # type: ignore
        from peft import PeftModel  # type: ignore

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        hf_logging.set_verbosity_error()

        resolved_model_name = self.model_name
        model_name_path = Path(self.model_name)
        if not model_name_path.is_absolute() and not model_name_path.exists():
            alt_path = Path("tiny-llm") / model_name_path
            if alt_path.exists():
                resolved_model_name = str(alt_path)

        model_source = resolved_model_name
        tokenizer_source = resolved_model_name
        adapter_source = None
        model_path = Path(resolved_model_name)
        if (
            model_path.exists()
            and model_path.is_dir()
            and (model_path / "adapter_config.json").exists()
            and (model_path / "adapter_model.safetensors").exists()
            and not (model_path / "config.json").exists()
        ):
            adapter_source = str(model_path)
            adapter_cfg = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
            base_model = str(adapter_cfg.get("base_model_name_or_path", "")).strip()
            if not base_model:
                raise RuntimeError(f"LoRA adapter at {model_path} is missing base_model_name_or_path in adapter_config.json")
            model_source = base_model
            if not ((model_path / "tokenizer.json").exists() or (model_path / "tokenizer.model").exists()):
                tokenizer_source = base_model

        try:
            self._tok = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        except Exception:
            self._tok = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

        dtype = torch.float16 if self._device == "cuda" else torch.float32
        if self._device == "cuda":
            try:
                self._model = self._from_pretrained_with_dtype(model_source, dtype=dtype, device_map="auto")
            except ValueError as e:
                if "requires `accelerate`" not in str(e):
                    raise
                self._model = self._from_pretrained_with_dtype(model_source, dtype=dtype, device_map=None)
                self._model.to("cuda")
        else:
            self._model = self._from_pretrained_with_dtype(model_source, dtype=dtype, device_map=None)
            self._model.to("cpu")
        if adapter_source:
            self._model = PeftModel.from_pretrained(self._model, adapter_source)
        self._model.eval()

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.use_chat_template and hasattr(self._tok, "apply_chat_template"):
            return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = []
        for m in messages:
            role_raw = m.get("role", "user").strip().lower()
            if role_raw == "system":
                role = "System"
            elif role_raw == "assistant":
                role = "Assistant"
            else:
                role = "User"
            content = m.get("content", "")
            out.append(f"{role}: {content}")
        out.append("Assistant:")
        return "\n\n".join(out)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        self._lazy_load()
        prompt = self._build_prompt(messages)
        inputs = self._tok(prompt, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        do_sample = self.temperature > 1e-6
        if not do_sample:
            try:
                self._model.generation_config.top_k = None
                self._model.generation_config.top_p = None
                self._model.generation_config.temperature = None
            except Exception:
                pass
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max(16, self.max_new_tokens),
                do_sample=do_sample,
                temperature=max(0.01, self.temperature) if do_sample else None,
                top_p=0.95 if do_sample else None,
                eos_token_id=self._tok.eos_token_id,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out_ids[0][inputs["input_ids"].shape[1] :]
        text = self._tok.decode(gen, skip_special_tokens=True)
        return text.strip()


class LocalLLM:
    def __init__(
        self,
        backend: str = "hf",
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        tiny_ckpt: str = "checkpoints_v2/final.pt",
        tiny_tokenizer: str = "tokenizer.model",
        tiny_lora: str = "",
        tiny_top_p: float = 1.0,
        use_chat_template: bool = True,
    ) -> None:
        b = (backend or "hf").strip().lower()
        if b == "tiny":
            self._impl = TinyLocalLLM(
                ckpt_path=tiny_ckpt,
                tokenizer_path=tiny_tokenizer,
                lora_adapter_path=tiny_lora,
                temperature=temperature,
                top_p=tiny_top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            self._impl = HFLocalLLM(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_chat_template=bool(use_chat_template),
            )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        return self._impl.generate(messages)
