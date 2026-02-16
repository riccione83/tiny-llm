# Code + Review Training README (3B, 16GB VRAM)

This runbook contains the minimum commands needed to train a model focused on:
- code generation
- code reviews (bug finding, refactoring, test suggestions)

Honest note:
- Base continued pretraining (CPT) alone improves coding knowledge/style.
- To reach instruct-level chat/review behavior, you will usually also need a targeted SFT pass.

## 0) Setup

```powershell
cd tiny-llm
python -m pip install -U pip
python -m pip install -r ..\requirements.txt
```

## 1) Download a 3B base model

If you want a strict base-only workflow, use a non-instruct base model when available.
If you start from Qwen instruct, training is still valid, but it is not a pure base starting point.

```powershell
cd tiny-llm
python .\01_download_base.py --model_id Qwen/Qwen2.5-3B-Instruct --output_dir models/base_3b --dtype float16
```

Expected output:
- `tiny-llm/models/base_3b`

## 2) Fast online CPT on code (HF, non-gated)

This command uses `codeparrot/github-code` and filters Python + TypeScript.

```powershell
cd tiny-llm
python .\02_train_base.py `
  --model_dir models\base_3b `
  --output_dir models\base_3b_code_fast_16gb_v1 `
  --disable_local_data `
  --hf_source "codeparrot/github-code||train|code|800000" `
  --hf_code_languages "python,typescript" `
  --hf_require_language_tag `
  --repeat_sources `
  --max_steps 6000 `
  --learning_rate 2e-5 `
  --warmup_ratio 0.02 `
  --per_device_batch_size 2 `
  --grad_accum 12 `
  --block_size 768 `
  --auto_tune_shape `
  --auto_tune_batch_candidates "1,2,3" `
  --auto_tune_block_candidates "512,768,1024" `
  --gradient_checkpointing `
  --dtype float16 `
  --logging_steps 20 `
  --save_steps 500 `
  --save_total_limit 4 `
  --disable_sample_logging
```

Expected output:
- `tiny-llm/models/base_3b_code_fast_16gb_v1/checkpoint-*`
- final model in `tiny-llm/models/base_3b_code_fast_16gb_v1`

## 3) Resume after interruption

```powershell
cd tiny-llm
$latest = (Get-ChildItem models/base_3b_code_fast_16gb_v1 -Directory | Where-Object { $_.Name -match '^checkpoint-\d+$' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

python .\02_train_base.py `
  --model_dir models\base_3b `
  --output_dir models\base_3b_code_fast_16gb_v1 `
  --disable_local_data `
  --hf_source "codeparrot/github-code||train|code|800000" `
  --hf_code_languages "python,typescript" `
  --hf_require_language_tag `
  --repeat_sources `
  --max_steps 6000 `
  --learning_rate 2e-5 `
  --warmup_ratio 0.02 `
  --per_device_batch_size 2 `
  --grad_accum 12 `
  --block_size 768 `
  --auto_tune_shape `
  --auto_tune_batch_candidates "1,2,3" `
  --auto_tune_block_candidates "512,768,1024" `
  --gradient_checkpointing `
  --dtype float16 `
  --logging_steps 20 `
  --save_steps 500 `
  --save_total_limit 4 `
  --disable_sample_logging `
  --resume_from_checkpoint "$latest"
```

## 4) Quick post-CPT manual checks

Recommended prompts:
1. "Write a TypeScript function `debounce(fn, ms)` with tests."
2. "Review this Python function and list bugs, complexity risks, and missing tests: ..."
3. "Refactor this code while preserving behavior and explain tradeoffs briefly: ..."

## 5) To get strong code-review capability (recommended)

CPT-only is usually not enough for robust chat/review behavior.
Recommended minimal pipeline:
1. CPT (step 2)
2. Short SFT on review-focused data (`04_train_lora.py`) with examples:
- input: patch/diff/snippet
- output: severity-ordered findings, concrete fixes, missing tests

Fast seed SFT example:

```powershell
cd tiny-llm
python .\04_train_lora.py `
  --model_dir models/base_3b_code_fast_16gb_v1 `
  --output_dir models/lora3b_code_review_seed_v1 `
  --disable_hf_data `
  --local_jsonl_glob samples/sft/repair_math_logic_coding.jsonl `
  --local_jsonl_glob samples/sft/format_constraints_strict.jsonl `
  --max_steps 300 `
  --max_length 1024 `
  --per_device_batch_size 1 `
  --grad_accum 16 `
  --learning_rate 5e-5 `
  --gradient_checkpointing `
  --dtype float16 `
  --logging_steps 20 `
  --save_steps 100 `
  --save_total_limit 6
```

## 6) 16GB troubleshooting

OOM:
- lower `--auto_tune_block_candidates` to `"512,768"`
- or increase `--grad_accum` (for example 16/24)
- or set `--per_device_batch_size 1`

Dataset too slow:
- lower `max_texts` in `--hf_source`
- disable preview/sample logging

## 7) Realistic expectations

- After fast CPT: better coding vocabulary/patterns.
- To get close to Qwen 3B on review chat: you need review-specific data + SFT + continuous evaluation.
- On a narrow domain (your stack, your standards), you can beat a general-purpose Qwen 3B.
