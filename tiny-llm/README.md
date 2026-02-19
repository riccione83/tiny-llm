# tiny-llm 7B (Official Minimal Runbook)

This repository is now centered on the **7B workflow**.
This README contains only the required steps to build and release the 7B model from scratch.

Legacy/alternative runbooks (3B, advanced stack, code-review training) were moved to `tiny-llm/docs/archive/`.

## 1) Install

```powershell
cd tiny-llm
python -m pip install -U pip
python -m pip install -r ..\requirements.txt
python -m pip install bitsandbytes
```

## 2) Download 7B Base

```powershell
cd tiny-llm
python .\03_download_lora_base.py --model_id Qwen/Qwen2.5-7B-Instruct --output_dir models/lora_base_7b --dtype float16
```

## 3) Train 7B Seed LoRA (QLoRA)

```powershell
cd tiny-llm
python .\04_train_lora.py `
  --model_dir models/lora_base_7b `
  --output_dir models/lora7b_seed_v1 `
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
  --learning_rate 3e-5 `
  --gradient_checkpointing `
  --dtype float16 `
  --use_4bit `
  --bnb_4bit_quant_type nf4 `
  --bnb_4bit_compute_dtype float16 `
  --logging_steps 20 `
  --save_steps 100 `
  --save_total_limit 6
```

## 4) Evaluate Checkpoints

```powershell
cd tiny-llm
python .\05_eval_lora_checkpoints.py --base_model_dir models/lora_base_7b --adapter_dir models/lora7b_seed_v1 --max_checkpoints 6 --device_map cuda --out_json models/lora7b_seed_v1/checkpoint_eval_report.json
```

Pick the best checkpoint from the scoreboard (example: `checkpoint-300`).

## 5) Regression Suite on Selected Checkpoint

```powershell
cd tiny-llm
python .\regression_suite.py --backend hf --model_dir models/lora_base_7b --adapter_dir models/lora7b_seed_v1/checkpoint-300 --device cuda --chat_format tokenizer --max_new_tokens 120
```

## 6) Release to LM Studio

```powershell
cd tiny-llm
.\release_lmstudio.ps1 `
  -BaseModelDir models/lora_base_7b `
  -AdapterDir models/lora7b_seed_v1 `
  -Checkpoint checkpoint-300 `
  -EvalReport models/lora7b_seed_v1/checkpoint_eval_report.json `
  -ReleaseName tyny-lm-7b-release1 `
  -QuantType Q8_0
```

## 7) LM Studio Settings

- Prompt Template: model default (`Auto`), not `Empty`
- Temperature for compliance tests: `0.0` to `0.2`

## 8) Quick Smoke Test

1. `Write a Python function is_prime(n). Return only a fenced python block.`
2. `Return ONLY valid JSON {"language": string, "has_code": boolean, "code": string}. Task: write add(a,b).`
3. `Give exactly 3 bullet points on risks of fine-tuning with small datasets.`
4. `Compute 47*19. Reply with: "<number>. <one short sentence>".`
