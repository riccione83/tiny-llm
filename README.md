# Mini Assistant (Grounded Web QA)

This repo supports two official paths:

- `mini_assistant/` (recommended): grounded QA with web retrieval.
- `tiny-llm/`: full local end-to-end training pipeline.

Use `mini_assistant/` if you want immediate quality and simpler setup.
Use `tiny-llm/` if you want to train everything locally from scratch.

## tiny-llm Status

The `tiny-llm` pipeline is now centered on the **7B workflow** (`Qwen/Qwen2.5-7B-Instruct` + LoRA/QLoRA).
Use `tiny-llm/README.md` for the official minimal 7B runbook.

## Path A (Recommended): Ready Model + Grounded QA

- LLM: `Qwen/Qwen3-4B-Instruct-2507`
- Retriever embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Runtime package: `mini_assistant/`
- Optional local backend: your custom tiny checkpoint from `legacy_chat.py` (legacy custom model backend)

## Quick Start

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Run chat (search web automatically):

```powershell
python -m mini_assistant.chat --backend hf --model_name Qwen/Qwen3-4B-Instruct-2507 --embedding_model sentence-transformers/all-MiniLM-L6-v2 --temperature 0.0
```

Show routing/debug (`direct` vs `web`):

```powershell
python -m mini_assistant.chat --show_debug --direct_confidence_threshold 0.72
```

Run chat on a fixed URL:

```powershell
python -m mini_assistant.chat --backend hf --url https://en.wikipedia.org/wiki/Italy
```

## Path B: Local End-to-End Training (No External Base LLM)

Training is now a modern minimal 4-script pipeline in `tiny-llm/`:

```powershell
cd tiny-llm
python .\01_download_base.py --model_id Qwen/Qwen2.5-0.5B-Instruct --output_dir models/base
python .\02_train_base.py --model_dir models/base --output_dir models/base_trained --recipe knowledge-heavy --max_steps 30000 --repeat_sources --gradient_checkpointing
python .\03_download_lora_base.py --model_id Qwen/Qwen3-4B-Instruct-2507 --output_dir models/lora_base
python .\04_train_lora.py --model_dir models/lora_base --output_dir models/lora_adapter --recipe heavy --max_steps 8000 --repeat_sources --gradient_checkpointing --save_merged
```

Production repair/release path (LM Studio ready):

```powershell
cd tiny-llm
.\run_lora_targeted_repair.ps1 -ModelDir models/base_trained -SeedAdapter models/lora_repair_v1/checkpoint-900 -OutDir models/lora_repair_v2
python .\05_eval_lora_checkpoints.py --base_model_dir models/base_trained --adapter_dir models/lora_repair_v2 --max_checkpoints 6 --out_json models/lora_repair_v2/checkpoint_eval_report.json
.\release_lmstudio.ps1 -BaseModelDir models/base_trained -AdapterDir models/lora_repair_v2 -ReleaseName tyny-lm-release2 -CleanupOldCheckpoints -CleanupOldLmStudioModels
```

Full details and options: `tiny-llm/README.md`.
Built-in local samples are in `tiny-llm/samples/` and are loaded by default in both training scripts.

Run regression eval:

```powershell
# Grounded (web QA) regression:
python .\eval.py --suite grounded --backend hf --model_name Qwen/Qwen3-4B-Instruct-2507 --embedding_model sentence-transformers/all-MiniLM-L6-v2

# Offline chat sanity (no web):
python .\eval.py --suite chat --backend hf --model_name Qwen/Qwen3-4B-Instruct-2507
```

Run confidence-gate check:

```powershell
python -m mini_assistant.eval_confidence_gate --backend hf --model_name Qwen/Qwen3-4B-Instruct-2507 --embedding_model sentence-transformers/all-MiniLM-L6-v2 --direct_confidence_threshold 0.72
```

## Notes

- The new entrypoint for practical usage is `mini_assistant/chat.py`.
- Best quality today is with `--backend hf` (`Qwen/Qwen3-4B-Instruct-2507`).
- `--backend tiny` is legacy/optional and requires you to provide your own custom checkpoint artifacts.

## What To Ask

These prompts are known-good for current setups.

Mini Assistant (`mini_assistant/chat.py`) examples:

- `What is the VRAM of NVIDIA RTX 5070 Ti? Use official sources only.`
- `What is the latest NVIDIA GPU line?`
- `/url https://en.wikipedia.org/wiki/Italy` then ask: `What is the official language?`
- `/url https://en.wikipedia.org/wiki/Italy` then ask: `Summarize the key points in 3 bullets.`

tiny-llm 7B (`tyny-lm-7b-release1` in LM Studio) examples:

- `Write a Python function is_prime(n). Return only a fenced python block.`
- `Return ONLY valid JSON {"language": string, "has_code": boolean, "code": string}. Task: write add(a,b).`
- `Give exactly 3 bullets on the risks of fine-tuning with small datasets.`
- `Compute 47*19. Reply with: "<number>. <one short sentence>".`
- `Write a TypeScript function quicksort(arr: number[]): number[]. Output only fenced ts.`
- `Remember this information. My name is Riccardo.` then: `What is my name?`
