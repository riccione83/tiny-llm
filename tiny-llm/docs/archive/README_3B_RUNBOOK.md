# tiny-llm 3B Runbook (16GB VRAM)

Operational runbook to move to a 3B base model with a robust LoRA pipeline (data validation, anti-drift guardrails, evaluation, LM Studio release).

For 7B QLoRA + RAG + router workflow, see `README_ADVANCED_STACK.md`.

## Goal

- Use a 3B base (`Qwen/Qwen2.5-3B-Instruct`) with the current pipeline.
- Avoid silent regressions in formatting/style behavior.
- Use restart-safe commands for long runs.

## Prerequisites

```powershell
cd tiny-llm
python -m pip install -U pip
python -m pip install -r ..\requirements.txt
```

Practical notes:
- VRAM: 16GB (sufficient for this runbook).
- Disk space: at least 40-60GB free.
- Use a dedicated shell; avoid closing it during long training jobs.

## Estimated Time

- Download 3B base: 10-40 min (network dependent).
- Seed LoRA (300 steps): 2-8 hours (GPU dependent).
- Checkpoint eval: 10-40 min.
- Conservative repair (120 steps): 1-4 hours.
- Merge + GGUF + release: 10-40 min.

## Step 1 - Download Base 3B

```powershell
cd tiny-llm
python .\03_download_lora_base.py --model_id Qwen/Qwen2.5-3B-Instruct --output_dir models/lora_base_3b --dtype float16
```

Expected output:
- `models/lora_base_3b`

## Step 2 - Seed LoRA (from scratch)

```powershell
cd tiny-llm
python .\04_train_lora.py `
  --model_dir models/lora_base_3b `
  --output_dir models/lora3b_seed_v1 `
  --disable_hf_data `
  --validate_data `
  --chat_format tokenizer `
  --code_fence_hygiene normalize `
  --reject_no_markdown_code_examples `
  --fail_on_duplicate_examples `
  --max_duplicate_example_ratio 0.10 `
  --min_loaded_examples 250 `
  --local_jsonl_glob samples/sft/repair_math_logic_coding.jsonl `
  --local_jsonl_glob samples/sft/system_styles.jsonl `
  --local_jsonl_glob samples/sft/chat_alignment_samples.jsonl `
  --local_jsonl_glob samples/sft/formatting_code_fences.jsonl `
  --local_jsonl_glob samples/sft/format_constraints_strict.jsonl `
  --local_jsonl_glob samples/sft/math_reasoning_micro.jsonl `
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

Expected output:
- `models/lora3b_seed_v1/checkpoint-*`

## Step 2B - Resume After Interruption

Find the latest checkpoint:

```powershell
cd tiny-llm
$latest = (Get-ChildItem models/lora3b_seed_v1 -Directory | Where-Object { $_.Name -match '^checkpoint-\d+$' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$latest
```

Resume from that checkpoint:

```powershell
cd tiny-llm
python .\04_train_lora.py `
  --model_dir models/lora_base_3b `
  --output_dir models/lora3b_seed_v1 `
  --disable_hf_data `
  --validate_data `
  --chat_format tokenizer `
  --code_fence_hygiene normalize `
  --reject_no_markdown_code_examples `
  --fail_on_duplicate_examples `
  --max_duplicate_example_ratio 0.10 `
  --min_loaded_examples 250 `
  --local_jsonl_glob samples/sft/repair_math_logic_coding.jsonl `
  --local_jsonl_glob samples/sft/system_styles.jsonl `
  --local_jsonl_glob samples/sft/chat_alignment_samples.jsonl `
  --local_jsonl_glob samples/sft/formatting_code_fences.jsonl `
  --local_jsonl_glob samples/sft/format_constraints_strict.jsonl `
  --local_jsonl_glob samples/sft/math_reasoning_micro.jsonl `
  --max_steps 300 `
  --max_length 1024 `
  --per_device_batch_size 1 `
  --grad_accum 16 `
  --learning_rate 5e-5 `
  --gradient_checkpointing `
  --dtype float16 `
  --logging_steps 20 `
  --save_steps 100 `
  --save_total_limit 6 `
  --resume_from_checkpoint "$latest"
```

## Step 3 - Evaluate Seed Checkpoints

```powershell
cd tiny-llm
python .\05_eval_lora_checkpoints.py --base_model_dir models/lora_base_3b --adapter_dir models/lora3b_seed_v1 --max_checkpoints 6 --out_json models/lora3b_seed_v1/checkpoint_eval_report.json
```

Expected output:
- `models/lora3b_seed_v1/checkpoint_eval_report.json`

## Step 4 - Conservative Repair (Optional, Recommended)

Replace `checkpoint-XXX` with your selected best checkpoint.

```powershell
cd tiny-llm
.\run_lora_targeted_repair.ps1 `
  -ModelDir models/lora_base_3b `
  -SeedAdapter models/lora3b_seed_v1/checkpoint-XXX `
  -OutDir models/lora3b_repair_v1 `
  -MaxSteps 120 `
  -LearningRate 8e-6 `
  -DType float16 `
  -MinLoadedExamples 250 `
  -MaxDuplicateExampleRatio 0.10
```

Eval repair:

```powershell
cd tiny-llm
python .\05_eval_lora_checkpoints.py --base_model_dir models/lora_base_3b --adapter_dir models/lora3b_repair_v1 --max_checkpoints 6 --out_json models/lora3b_repair_v1/checkpoint_eval_report.json
```

## Step 5 - Release LM Studio

Replace `checkpoint-YYY` with the promoted checkpoint.

```powershell
cd tiny-llm
.\release_lmstudio.ps1 `
  -BaseModelDir models/lora_base_3b `
  -AdapterDir models/lora3b_repair_v1 `
  -Checkpoint checkpoint-YYY `
  -EvalReport models/lora3b_repair_v1/checkpoint_eval_report.json `
  -ReleaseName tyny-lm-3b-release1 `
  -QuantType Q8_0
```

### Freeze As Stable

Once validated, freeze the same checkpoint under a stable release name:

```powershell
cd tiny-llm
.\release_lmstudio.ps1 `
  -BaseModelDir models/lora_base_3b `
  -AdapterDir models/lora3b_repair_v1 `
  -Checkpoint checkpoint-420 `
  -EvalReport models/lora3b_repair_v1/checkpoint_eval_report.json `
  -ReleaseName tyny-lm-3b-stable `
  -QuantType Q8_0
```

Notes:
- `Q8_0` = higher quality.
- `Q4_K_M` = lighter/faster.

## Step 6 - LM Studio Setup (Strict Testing)

- Prompt Template: model default/Auto (for Qwen-family, ChatML is correct).
- Do NOT use `Empty` (it bypasses role formatting and often causes style/format drift).
- Temperature: `0.0` or `0.2` for compliance testing.
- Top-p: `0.9`
- Min-p: off
- Repeat penalty: `1.05`
- Always start a new chat before smoke tests.

## Minimum Smoke Test (Post-Release)

1. `Write a Python function is_prime(n). Return only a fenced python block.`
2. `Give exactly 3 bullet points on risks of fine-tuning with small datasets.`
3. `Compute 47*19. Reply with: "<number>. <one short sentence>".`
4. `Return ONLY valid JSON {"language": string, "has_code": boolean, "code": string}. Task: write add(a,b).`

## Common Errors

- `Output dir already contains checkpoints`:
  - use a new `-OutDir`, or use `-AllowExistingOutDir` only if you explicitly accept drift risk.
- OOM/CUDA memory:
  - reduce `--max_length` (for example `768`) or increase `--grad_accum` while keeping batch size 1.
- Responses do not follow requested format in LM Studio:
  - verify Prompt Template is `Auto/default` (or `ChatML` for Qwen-family), not `Empty`.
  - rerun with `temperature=0.0`.
  - run each smoke test in a fresh chat (except explicit multi-turn tests).
