# tiny-llm (Training + Repair + LM Studio Release)

This folder contains the full local pipeline to train, repair, evaluate, and ship a GGUF model to LM Studio.

## Install

```powershell
python -m pip install -U pip
python -m pip install -r ..\requirements.txt
```

## Minimal Runbook (Existing Base + Seed Adapter)

Use this when you already have `models/base_trained` and a seed adapter checkpoint.

```powershell
cd tiny-llm
.\run_lora_targeted_repair.ps1 -ModelDir models/base_trained -SeedAdapter models/lora_repair_v1/checkpoint-900 -OutDir models/lora_repair_v2
python .\05_eval_lora_checkpoints.py --base_model_dir models/base_trained --adapter_dir models/lora_repair_v2 --max_checkpoints 6 --out_json models/lora_repair_v2/checkpoint_eval_report.json
.\release_lmstudio.ps1 -BaseModelDir models/base_trained -AdapterDir models/lora_repair_v2 -ReleaseName tyny-lm-release2 -CleanupOldCheckpoints -CleanupOldLmStudioModels
```

Output in LM Studio:
- `C:\Users\<you>\.lmstudio\models\<you>\tyny-lm-release2\tyny-lm-release2-q8_0.gguf`

## Cold Start (No Local Artifacts)

Use this when starting from an empty `tiny-llm/models`.

```powershell
cd tiny-llm
python .\01_download_base.py --model_id Qwen/Qwen2.5-0.5B-Instruct --output_dir models/base
python .\02_train_base.py --model_dir models/base --output_dir models/base_trained --recipe knowledge-heavy --max_steps 30000 --repeat_sources --gradient_checkpointing
.\run_lora_sft_quality.ps1 -ModelDir models/base_trained -OutDir models/lora_repair_v1
.\run_lora_targeted_repair.ps1 -ModelDir models/base_trained -SeedAdapter models/lora_repair_v1/checkpoint-900 -OutDir models/lora_repair_v2
python .\05_eval_lora_checkpoints.py --base_model_dir models/base_trained --adapter_dir models/lora_repair_v2 --max_checkpoints 6 --out_json models/lora_repair_v2/checkpoint_eval_report.json
.\release_lmstudio.ps1 -BaseModelDir models/base_trained -AdapterDir models/lora_repair_v2 -ReleaseName tyny-lm-release2 -CleanupOldCheckpoints -CleanupOldLmStudioModels
```

## Automation Scripts

- `run_lora_targeted_repair.ps1`: conservative targeted repair with strict JSONL validation and code-fence hygiene.
- `05_eval_lora_checkpoints.py`: scores recent checkpoints on a fixed prompt set.
- `release_lmstudio.ps1`: selects best checkpoint from eval report (fallback: latest), merges LoRA, converts to GGUF, verifies chat-template metadata, quantizes, deploys to LM Studio, optional cleanup.
- `06_merge_lora_checkpoint.py`: helper used by release script to merge a specific checkpoint into the base model.
- `07_verify_gguf_chat_template.py`: validates that GGUF includes a non-empty `tokenizer.chat_template` and can compare it with the base tokenizer.

## Data Safety Defaults

`04_train_lora.py` now supports:
- strict JSONL validation with fail-fast errors (`--validate_data`)
- append-mode local data globs (`--local_jsonl_glob` repeated)
- minimum post-filter example guard (`--min_loaded_examples`, override via `--allow_small_dataset`)
- duplicate example guard (`--fail_on_duplicate_examples`, `--max_duplicate_example_ratio`)
- code-fence hygiene (`--code_fence_hygiene normalize|reject`)
- chat format alignment (`--chat_format tokenizer|legacy|auto`)

Targeted format-repair dataset:
- `samples/sft/format_constraints_strict.jsonl` (fenced python, strict JSON, exact bullets, one-sentence, math format, injection resistance)

## Quick Regression Check

```powershell
cd tiny-llm
python .\regression_suite.py --backend mock
```

## LM Studio Notes

- Keep Prompt Template on model default (or `Empty`): do not force a different model template.
- For strict-format checks, start with lower randomness (`temperature` ~ `0.2`).
