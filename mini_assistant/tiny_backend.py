import importlib.util
import os
from pathlib import Path
from typing import Dict, List, Optional

import sentencepiece as spm
import torch


def _load_legacy_module(path: str = "09_chat.py"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing legacy tiny model file: {p}")
    spec = importlib.util.spec_from_file_location("tiny_legacy_chat", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TinyLocalLLM:
    def __init__(
        self,
        ckpt_path: str,
        tokenizer_path: str,
        lora_adapter_path: str = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 160,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.tokenizer_path = tokenizer_path
        self.lora_adapter_path = lora_adapter_path
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._legacy = None
        self._model = None
        self._sp = None

    def _lazy_load(self) -> None:
        if self._legacy is not None and self._model is not None and self._sp is not None:
            return
        self._legacy = _load_legacy_module("09_chat.py")
        self._legacy.MAX_NEW_TOKENS = max(16, self.max_new_tokens)

        sp = spm.SentencePieceProcessor()
        if not sp.load(self.tokenizer_path):
            raise RuntimeError(f"Unable to load tokenizer: {self.tokenizer_path}")
        vocab = sp.get_piece_size()

        model = self._legacy.FastGPT(
            vocab=vocab,
            dim=self._legacy.EMBED_DIM,
            heads=self._legacy.NUM_HEADS,
            layers=self._legacy.NUM_LAYERS,
            block_size=self._legacy.BLOCK_SIZE,
        ).to(self._device)

        ckpt = torch.load(self.ckpt_path, map_location=self._device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

        if self.lora_adapter_path:
            if not Path(self.lora_adapter_path).exists():
                raise FileNotFoundError(f"LoRA adapter not found: {self.lora_adapter_path}")
            self._legacy.inject_lora(model, self._device)
            lora_sd = torch.load(self.lora_adapter_path, map_location="cpu")
            self._legacy.load_lora_state_dict(model, lora_sd, self._device)

        model.eval()
        self._model = model
        self._sp = sp

    @staticmethod
    def _build_prompt(messages: List[Dict[str, str]]) -> str:
        out: List[str] = []
        for m in messages:
            role = m.get("role", "user").strip().lower()
            content = m.get("content", "").strip()
            if role == "system":
                out.append(f"System: {content}")
            elif role == "assistant":
                out.append(f"Assistant: {content}")
            else:
                out.append(f"User: {content}")
        out.append("Assistant:")
        return "\n\n".join(out).strip()

    def generate(self, messages: List[Dict[str, str]]) -> str:
        self._lazy_load()
        prompt = self._build_prompt(messages)
        ans = self._legacy.generate(
            model=self._model,
            sp=self._sp,
            device=self._device,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            return_meta=False,
        )
        return (ans or "").strip()

