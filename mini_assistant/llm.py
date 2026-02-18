from typing import Dict, List

import os
from pathlib import Path
import json
import time
import threading
import torch

from .tiny_backend import TinyLocalLLM


class HFLocalLLM:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        use_chat_template: bool = True,
        quantization: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.use_chat_template = bool(use_chat_template)
        self.quantization = str(quantization or "auto").strip().lower()
        self._tok = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._last_profile: Dict[str, float | int | str] = {}
        self._last_stream_pieces: List[str] = []
        self._gen_lock = threading.Lock()
        self._context_window_tokens = 0
        self._quant_mode_effective = "none"

    @staticmethod
    def _from_pretrained(model_name: str, dtype: torch.dtype, device_map, quantization_config=None):
        from transformers import AutoModelForCausalLM  # type: ignore

        attn_impl = os.environ.get("TINYLLM_ATTN_IMPL", "").strip().lower()
        if not attn_impl:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "flash_sdp_enabled"):
                # Prefer fast SDPA kernels on recent CUDA stacks.
                attn_impl = "sdpa"
            else:
                attn_impl = "eager"
        kwargs = {
            "device_map": device_map,
            "attn_implementation": attn_impl,
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        try:
            if quantization_config is None:
                return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except TypeError:
            if quantization_config is None:
                return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def _resolve_quantization(self):
        mode = self.quantization
        if mode not in {"auto", "none", "4bit", "8bit"}:
            mode = "auto"
        if self._device != "cuda":
            return None, "none"
        if mode == "auto":
            # Keep smaller models unquantized by default and prefer 4-bit for larger models.
            needle = self.model_name.lower()
            large_markers = ("7b", "8b", "13b", "14b", "32b", "34b", "70b")
            mode = "4bit" if any(m in needle for m in large_markers) else "none"
        if mode == "none":
            return None, "none"
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            import importlib.util

            if importlib.util.find_spec("bitsandbytes") is None:
                raise RuntimeError("bitsandbytes package is not installed")
            bf16_ok = False
            try:
                bf16_ok = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16_ok = False
            compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
            if mode == "8bit":
                qcfg = BitsAndBytesConfig(load_in_8bit=True)
                return qcfg, "8bit"
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            return qcfg, "4bit"
        except Exception as e:
            print(f"[HFLocalLLM] Quantization '{mode}' unavailable, using full precision: {e}")
            return None, "none"

    def _infer_context_window_tokens(self) -> int:
        values: List[int] = []
        cfg = getattr(self._model, "config", None)
        for key in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len", "model_max_length"):
            try:
                v = int(getattr(cfg, key))
                if v > 0:
                    values.append(v)
            except Exception:
                pass
        try:
            rope = getattr(cfg, "rope_scaling", None)
            if isinstance(rope, dict):
                factor = float(rope.get("factor", 1.0))
                orig = int(
                    rope.get("original_max_position_embeddings", 0)
                    or getattr(cfg, "max_position_embeddings", 0)
                    or 0
                )
                if factor > 1.0 and orig > 0:
                    values.append(int(orig * factor))
        except Exception:
            pass
        try:
            tok_max = int(getattr(self._tok, "model_max_length", 0))
            # Ignore "very large sentinel" values often used by tokenizers.
            if 0 < tok_max < 1_000_000:
                values.append(tok_max)
        except Exception:
            pass
        if not values:
            return 0
        return min(values)

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
            base_model_path = Path(base_model)
            if not base_model_path.is_absolute() and not base_model_path.exists():
                alt_base = Path("tiny-llm") / base_model_path
                if alt_base.exists():
                    model_source = str(alt_base)
            if not ((model_path / "tokenizer.json").exists() or (model_path / "tokenizer.model").exists()):
                tokenizer_source = model_source

        try:
            self._tok = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        except Exception:
            self._tok = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

        if self._device == "cuda":
            bf16_ok = False
            try:
                bf16_ok = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16_ok = False
            dtype = torch.bfloat16 if bf16_ok else torch.float16
            tf32_on = os.environ.get("TINYLLM_TF32", "1").strip().lower() in {"1", "true", "yes"}
            try:
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = bool(tf32_on)
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = bool(tf32_on)
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high" if tf32_on else "highest")
            except Exception:
                pass
            flash_only = os.environ.get("TINYLLM_SDPA_FLASH_ONLY", "0").strip().lower() in {"1", "true", "yes"}
            if flash_only:
                try:
                    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                        torch.backends.cuda.enable_flash_sdp(True)
                    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                        torch.backends.cuda.enable_mem_efficient_sdp(False)
                    if hasattr(torch.backends.cuda, "enable_math_sdp"):
                        torch.backends.cuda.enable_math_sdp(False)
                except Exception as e:
                    print(f"[HFLocalLLM] flash-only SDPA setup skipped: {e}")
        else:
            dtype = torch.float32
        quantization_config, quant_mode_effective = self._resolve_quantization()
        self._quant_mode_effective = str(quant_mode_effective)
        if self._device == "cuda":
            if quantization_config is not None:
                try:
                    self._model = self._from_pretrained(
                        model_source,
                        dtype=dtype,
                        device_map="auto",
                        quantization_config=quantization_config,
                    )
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Failed to load quantized model ({self._quant_mode_effective}) on CUDA: {e}"
                    ) from e
            else:
                # Prefer a full-GPU load for stability.
                # Auto device_map can leave mixed meta/cpu states that are brittle
                # for long-running API usage on Windows.
                try:
                    self._model = self._from_pretrained(model_source, dtype=dtype, device_map=None)
                    self._model.to("cuda")
                except RuntimeError as e:
                    msg = str(e).lower()
                    is_oom = ("out of memory" in msg) or ("cuda out of memory" in msg)
                    allow_auto_fallback = os.environ.get("TINYLLM_ALLOW_AUTO_DEVICE_MAP_FALLBACK", "0").strip() in {"1", "true", "yes"}
                    if is_oom and allow_auto_fallback:
                        self._model = self._from_pretrained(model_source, dtype=dtype, device_map="auto")
                    else:
                        raise RuntimeError(
                            "Failed to load model fully on CUDA. "
                            "Set TINYLLM_ALLOW_AUTO_DEVICE_MAP_FALLBACK=1 to allow auto offload fallback "
                            "(may be unstable on some Windows setups). "
                            f"Original error: {e}"
                        ) from e
        else:
            self._model = self._from_pretrained(model_source, dtype=dtype, device_map=None)
            self._model.to("cpu")
        if adapter_source:
            peft_kwargs = {}
            try:
                device_map = getattr(self._model, "hf_device_map", None)
                needs_offload = bool(device_map) and len(set(device_map.values()).intersection({"cpu", "disk"})) > 0
            except Exception:
                needs_offload = False
            if needs_offload:
                offload_dir = str(Path(adapter_source) / "_offload")
                Path(offload_dir).mkdir(parents=True, exist_ok=True)
                peft_kwargs["offload_dir"] = offload_dir
            self._model = PeftModel.from_pretrained(self._model, adapter_source, **peft_kwargs)
        # Guarantee generation cache is enabled for decode speed.
        try:
            self._model.config.use_cache = True
        except Exception:
            pass
        self._model.eval()
        if self._device == "cuda" and os.environ.get("TINYLLM_TORCH_COMPILE", "0").strip() in {"1", "true", "yes"}:
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"[HFLocalLLM] torch.compile skipped: {e}")

        # Detect accidental offload and keep a stable runtime profile.
        offload_targets = set()
        try:
            device_map = getattr(self._model, "hf_device_map", None)
            if isinstance(device_map, dict):
                offload_targets = {str(v) for v in device_map.values()}
        except Exception:
            offload_targets = set()
        if {"cpu", "disk"}.intersection(offload_targets):
            print(f"[HFLocalLLM] Warning: model is partially offloaded: {sorted(offload_targets)}")
        self._context_window_tokens = int(self._infer_context_window_tokens())
        if self._context_window_tokens > 0:
            print(
                f"[HFLocalLLM] Loaded model={self.model_name} "
                f"quant={self._quant_mode_effective} context={self._context_window_tokens}"
            )
        else:
            print(f"[HFLocalLLM] Loaded model={self.model_name} quant={self._quant_mode_effective}")

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
        with self._gen_lock:
            self._lazy_load()
            prompt = self._build_prompt(messages)
            t_tok0 = time.perf_counter()
            inputs = self._tok(prompt, return_tensors="pt")
            t_tok1 = time.perf_counter()
            prompt_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
            if self._device == "cuda":
                target_device = next(self._model.parameters()).device
                inputs = {k: v.to(target_device, non_blocking=True) for k, v in inputs.items()}
            do_sample = self.temperature > 1e-6
            if not do_sample:
                try:
                    self._model.generation_config.top_k = None
                    self._model.generation_config.top_p = None
                    self._model.generation_config.temperature = None
                except Exception:
                    pass
            t_gen0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=max(16, self.max_new_tokens),
                        do_sample=do_sample,
                        temperature=max(0.01, self.temperature) if do_sample else None,
                        top_p=0.95 if do_sample else None,
                        use_cache=True,
                        eos_token_id=self._tok.eos_token_id,
                        pad_token_id=self._tok.eos_token_id,
                    )
            except Exception as e:
                flash_only = os.environ.get("TINYLLM_SDPA_FLASH_ONLY", "0").strip().lower() in {"1", "true", "yes"}
                if (not flash_only) or self._device != "cuda":
                    raise
                msg = str(e).lower()
                flash_related = (
                    "no available kernel" in msg
                    or "flash attention" in msg
                    or "sdp" in msg
                )
                if not flash_related:
                    raise
                # Fallback path for unsupported flash kernels on some CUDA/PyTorch stacks.
                try:
                    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                        torch.backends.cuda.enable_mem_efficient_sdp(True)
                    if hasattr(torch.backends.cuda, "enable_math_sdp"):
                        torch.backends.cuda.enable_math_sdp(True)
                    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                        torch.backends.cuda.enable_flash_sdp(True)
                    with torch.inference_mode():
                        out_ids = self._model.generate(
                            **inputs,
                            max_new_tokens=max(16, self.max_new_tokens),
                            do_sample=do_sample,
                            temperature=max(0.01, self.temperature) if do_sample else None,
                            top_p=0.95 if do_sample else None,
                            use_cache=True,
                            eos_token_id=self._tok.eos_token_id,
                            pad_token_id=self._tok.eos_token_id,
                        )
                    print(f"[HFLocalLLM] Flash-only SDPA fallback engaged: {e}")
                except Exception:
                    raise
            if self._device == "cuda":
                torch.cuda.synchronize()
            t_gen1 = time.perf_counter()
            gen = out_ids[0][inputs["input_ids"].shape[1] :]
            text = self._tok.decode(gen, skip_special_tokens=True)
            # Token-like chunks for SSE clients.
            pieces: List[str] = []
            try:
                special_ids = set(getattr(self._tok, "all_special_ids", []) or [])
                for tid in gen.tolist():
                    if int(tid) in special_ids:
                        continue
                    frag = self._tok.decode([int(tid)], skip_special_tokens=False)
                    if frag:
                        pieces.append(frag)
            except Exception:
                pieces = []
            self._last_stream_pieces = pieces
            gen_tokens = int(gen.shape[0])
            gen_s = max(1e-9, (t_gen1 - t_gen0))
            tok_s = max(0.0, (t_tok1 - t_tok0))
            device = "cpu"
            dtype = "unknown"
            gpu_mem_mb = 0.0
            try:
                p0 = next(self._model.parameters())
                device = str(p0.device)
                dtype = str(p0.dtype).replace("torch.", "")
            except Exception:
                pass
            if self._device == "cuda":
                try:
                    gpu_mem_mb = float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
                except Exception:
                    gpu_mem_mb = 0.0
            self._last_profile = {
                "tokenize_ms": round(tok_s * 1000.0, 2),
                "generate_ms": round(gen_s * 1000.0, 2),
                "prompt_tokens": prompt_tokens,
                "tokens_generated": gen_tokens,
                "tokens_per_sec": round(gen_tokens / gen_s, 2),
                "device": device,
                "dtype": dtype,
                "quantization": self._quant_mode_effective,
                "gpu_mem_allocated_mb": round(gpu_mem_mb, 2),
            }
            return text.strip()

    def ensure_loaded(self) -> None:
        with self._gen_lock:
            self._lazy_load()

    def context_window_tokens(self) -> int:
        with self._gen_lock:
            self._lazy_load()
            return int(self._context_window_tokens)


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
        quantization: str = "auto",
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
                quantization=str(quantization),
            )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        return self._impl.generate(messages)

    def ensure_loaded(self) -> None:
        lazy = getattr(self._impl, "ensure_loaded", None)
        if callable(lazy):
            lazy()

    def context_window_tokens(self) -> int:
        fn = getattr(self._impl, "context_window_tokens", None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                return 0
        return 0
