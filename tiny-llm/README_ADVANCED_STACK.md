# tiny-llm Advanced Stack Runbook

This runbook covers the three next upgrades:

1. 7B QLoRA training path on 16GB VRAM.
2. Local RAG + conversation memory.
3. Local/cloud router with one command surface.

## 0) Prerequisites

```powershell
cd tiny-llm
python -m pip install -U pip
python -m pip install -r ..\requirements.txt
```

For QLoRA 4-bit you also need `bitsandbytes`:

```powershell
python -m pip install bitsandbytes
```

Note: `bitsandbytes` support on native Windows can be limited. If install/runtime fails, run the same commands in Linux/WSL.

## 1) 7B QLoRA (Minimal, conservative)

Download a 7B instruct base:

```powershell
cd tiny-llm
python .\03_download_lora_base.py --model_id Qwen/Qwen2.5-7B-Instruct --output_dir models/lora_base_7b --dtype float16
```

Train LoRA with 4-bit loading:

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

Evaluate checkpoints:

```powershell
cd tiny-llm
python .\05_eval_lora_checkpoints.py --base_model_dir models/lora_base_7b --adapter_dir models/lora7b_seed_v1 --max_checkpoints 6 --out_json models/lora7b_seed_v1/checkpoint_eval_report.json
```

## 2) RAG + Memory

Use `rag_memory_router.py` with local retrieval chunks and persisted chat memory:

```powershell
cd tiny-llm
python .\rag_memory_router.py `
  --interactive `
  --router local `
  --knowledge_glob "samples/base/*.txt" `
  --knowledge_glob "samples/sft/*.jsonl" `
  --memory_file models/chat_memory/dev_session.jsonl `
  --knowledge_top_k 3 `
  --show_trace
```

Behavior:
- Retrieves top lexical chunks from `--knowledge_glob`.
- Injects recent turns from `--memory_file`.
- Appends each turn to memory as JSONL.

## 3) Local/Cloud Router

Set cloud key and run router in auto mode:

```powershell
$env:OPENAI_API_KEY = "YOUR_KEY"
cd tiny-llm
python .\rag_memory_router.py `
  --interactive `
  --router auto `
  --local_url "http://127.0.0.1:1234/v1/chat/completions" `
  --local_model "local-model" `
  --cloud_url "https://api.openai.com/v1/chat/completions" `
  --cloud_model "gpt-4.1-mini" `
  --memory_file models/chat_memory/router_session.jsonl `
  --knowledge_glob "samples/base/*.txt" `
  --show_trace
```

Auto routing policy:
- Code/format-heavy prompts: local.
- Longer strategy/tradeoff/domain prompts: cloud.
- Missing cloud key: local fallback.

## Optional: QLoRA targeted repair script

You can reuse `run_lora_targeted_repair.ps1` with optional 4-bit flags:

```powershell
cd tiny-llm
.\run_lora_targeted_repair.ps1 `
  -ModelDir models/lora_base_7b `
  -SeedAdapter models/lora7b_seed_v1/checkpoint-300 `
  -OutDir models/lora7b_repair_v1 `
  -MaxSteps 120 `
  -LearningRate 8e-6 `
  -Use4Bit `
  -Bnb4BitQuantType nf4 `
  -Bnb4BitComputeDType float16
```

## Smoke prompts to compare local vs cloud

1. `Write a Python function is_prime(n). Return only a fenced python block.`
2. `Give exactly 3 bullet points on risks of fine-tuning with small datasets.`
3. `Return ONLY valid JSON {"language": string, "has_code": boolean, "code": string}. Task: write add(a,b).`
4. `Compare two rollout strategies for a risky migration and give tradeoffs.`
