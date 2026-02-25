# tiny-llm 7B (Official Runbook)

This runbook is the official path to build, evaluate, and release the 7B model.
Legacy/alternative runbooks (3B, advanced stack, code-review training) are archived in `tiny-llm/docs/archive/`.

## Scope and Audience

Use this guide if you want reproducible 7B fine-tuning and release artifacts.
If you only need a local runtime assistant, use the repository root quickstart first.

## Platform Matrix (Training/Release)

| Platform Profile | 7B Training | QLoRA 4-bit | Checkpoint Eval | Regression | LM Studio Release Script |
|---|---|---|---|---|---|
| Windows + NVIDIA CUDA | Yes | Yes | Yes | Yes | Yes |
| Linux + NVIDIA CUDA | Yes | Yes | Yes | Yes | No |
| macOS Apple Silicon (MPS/Metal) | Yes (LoRA) | No | Yes | Yes | No |
| CPU-only | Limited/slow | No | Limited/slow | Limited/slow | No |

Release packaging is Windows-only because `scripts/release_lmstudio.ps1` uses Windows-native tooling and workflow assumptions.

## Preflight (Recommended)

Minimum practical targets:
- GPU: NVIDIA CUDA with `>=16 GB` VRAM for smoother 7B QLoRA
- RAM: `>=32 GB`
- Free disk: `60-100 GB` (base model, checkpoints, merged model, GGUF)
- Time budget: hours from first download to release artifact

Quick checks:

```bash
python --version
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('mps_available=', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
```

On NVIDIA systems:

```bash
nvidia-smi
```

Expected speed (indicative, `max_steps=300`):

| Profile | 7B Train Time | Checkpoint Eval | Notes |
|---|---|---|---|
| CUDA 16GB+ | 1-4 h | 15-60 min | Recommended baseline |
| Apple Silicon 16-32GB | 3-10 h (LoRA) | 30-120 min | Use LoRA mode, no 4-bit |
| CPU-only | Often impractical | Very slow | Use runtime/eval path first |

## Base Training Time Budget (50GB Corpus, Indicative)

This table is for full base training workloads on a `~50 GB` text corpus (not LoRA adapter tuning).
It is calibrated from your baseline:
- `RTX 5070 Ti 16GB`: about `5 days` at continuous `24/7` execution

| System Profile | Relative Throughput vs RTX 5070 Ti 16GB | Estimated Wall-Clock for 50GB (24/7) | Notes |
|---|---|---|---|
| RTX 4090 24GB | 1.4x-2.0x | 2.5-3.5 days | Larger VRAM and higher sustained throughput |
| RTX 5080/4080 class 16GB | 0.9x-1.2x | 4-6 days | Close to baseline in many setups |
| RTX 5070 Ti 16GB | 1.0x | about 5 days | Baseline reference |
| RTX 4070 Ti/SUPER 16GB | 0.6x-0.85x | 6-9 days | More sensitivity to thermal/power limits |
| Apple Silicon Max (M2/M3 class) | 0.35x-0.55x | 9-14 days | Feasible for LoRA and smaller base runs |
| CPU-only (desktop/server) | 0.08x-0.20x | 25-60 days | Generally impractical for 50GB base training |

Assumptions for this table:
- single machine, no multi-GPU or distributed training
- clean training run without long pauses
- similar recipe family and sequence lengths
- storage and thermals do not bottleneck sustained throughput

Use these values for planning budget and scheduling, not as hard guarantees.

## Optional: Base Training Modes (`02_train_base.py`)

`02_train_base.py` supports:
- `--init_mode pretrained`: continued pretraining from existing weights (`from_pretrained`)
- `--init_mode scratch`: scratch initialization (random weights) from config + tokenizer (`from_config`)

Examples:

```bash
# Continued pretraining (0.5B example)
python ./02_train_base.py --init_mode pretrained --model_dir models/base --output_dir models/base_trained

# Scratch initialization (0.5B random weights)
python ./02_train_base.py --init_mode scratch --config_source Qwen/Qwen2.5-0.5B-Instruct --output_dir models/base_scratch

# Scratch initialization (3B random weights)
python ./02_train_base.py --init_mode scratch --config_source Qwen/Qwen2.5-3B-Instruct --output_dir models/base_3b_scratch_v1

# Scratch initialization (7B random weights)
python ./02_train_base.py --init_mode scratch --config_source Qwen/Qwen2.5-7B-Instruct --output_dir models/base_7b_scratch_v1
```

Quickstart action IDs for scratch-initialization presets:
- `tiny.train.base.05b.scratch`
- `tiny.train.base.3b.scratch`
- `tiny.train.base.3b.scratch.wiki`
- `tiny.train.base.7b.scratch`

## Scratch Run Milestones (Indicative)

For random-init base training, early outputs are expected to look weak for a long window.

| Stage | Typical Step Range | What You Usually See |
|---|---|---|
| Bootstrap | `0-2k` | High loss, repetitive tokens, unstable sample quality |
| Early stabilization | `2k-10k` | Loss becomes less noisy, still far from useful generations |
| First consistent gains | `10k-50k` | Fewer repetitions, better local coherence in short outputs |
| Long-run refinement | `50k+` | Slower but steady improvements; quality gains become incremental |

These ranges are hardware/corpus dependent and are not pass/fail thresholds.

## Scratch Stability and Recovery

- `OOM after auto-tune`: probe success does not guarantee full-step success when optimizer state is allocated. Reduce sequence length first (`--block_size`), then batch shape.
- `NaN divergence`: if you see `grad_norm: nan` or extreme loss spikes, stop immediately and resume from the most recent healthy checkpoint.
- `Checkpoint health`: a checkpoint can have valid `trainer_state.json` but corrupted model weights if NaN happened after save scheduling; prefer the latest checkpoint with stable recent logs.
- `Windows worker errors`: if multiprocessing/pickling errors appear, run with `--dataloader_num_workers 0`.
- `Allocator env var`: use `PYTORCH_ALLOC_CONF` (new name) instead of deprecated `PYTORCH_CUDA_ALLOC_CONF`.

Resume pattern:

```bash
python ./02_train_base.py \
  --resume_from_checkpoint models/<run_dir>/checkpoint-<step> \
  --ignore_data_skip
```

## 1) Install

```bash
cd tiny-llm
python -m pip install -U pip
python -m pip install -r ../requirements.txt
```

Optional on CUDA environments (for QLoRA workflows):

```bash
python -m pip install bitsandbytes
```

Expected output: dependencies install without errors.

If you are not on CUDA, `bitsandbytes` may be unnecessary; keep going with standard LoRA.

## 2) Download 7B Base

```bash
cd tiny-llm
python ./03_download_lora_base.py --model_id Qwen/Qwen2.5-7B-Instruct --output_dir models/lora_base_7b --dtype float16
```

Expected output: `models/lora_base_7b/` exists and includes model/tokenizer files.

## 3) Train 7B Seed Adapter

### Profile A: CUDA + QLoRA 4-bit (default)

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

### Profile B: no CUDA (LoRA fallback)

Use the same command as Profile A, but remove:
- `--use_4bit`
- `--bnb_4bit_quant_type nf4`
- `--bnb_4bit_compute_dtype float16`

Expected output: `models/lora7b_seed_v1/checkpoint-*` directories are created.

## 4) Evaluate Checkpoints

CUDA profile:

```bash
cd tiny-llm
python ./05_eval_lora_checkpoints.py --base_model_dir models/lora_base_7b --adapter_dir models/lora7b_seed_v1 --max_checkpoints 6 --device_map cuda --out_json models/lora7b_seed_v1/checkpoint_eval_report.json
```

Non-CUDA profile:

```bash
cd tiny-llm
python ./05_eval_lora_checkpoints.py --base_model_dir models/lora_base_7b --adapter_dir models/lora7b_seed_v1 --max_checkpoints 6 --device_map auto --out_json models/lora7b_seed_v1/checkpoint_eval_report.json
```

Expected output: `models/lora7b_seed_v1/checkpoint_eval_report.json` is created.
Pick the best checkpoint from the scoreboard (example: `checkpoint-300`).

## 5) Regression Suite on Selected Checkpoint

CUDA profile:

```bash
cd tiny-llm
python ./regression_suite.py --backend hf --model_dir models/lora_base_7b --adapter_dir models/lora7b_seed_v1/checkpoint-300 --device cuda --chat_format tokenizer --max_new_tokens 120
```

Non-CUDA profile:

```bash
cd tiny-llm
python ./regression_suite.py --backend hf --model_dir models/lora_base_7b --adapter_dir models/lora7b_seed_v1/checkpoint-300 --device auto --chat_format tokenizer --max_new_tokens 120
```

Expected output: regression checks complete without critical failures.

## 6) Release to LM Studio (Windows Only)

```powershell
cd tiny-llm
.\scripts\release_lmstudio.ps1 `
  -BaseModelDir models/lora_base_7b `
  -AdapterDir models/lora7b_seed_v1 `
  -Checkpoint checkpoint-300 `
  -EvalReport models/lora7b_seed_v1/checkpoint_eval_report.json `
  -ReleaseName tiny-llm-7b-release1 `
  -QuantType Q8_0
```

Expected outputs:
- `models/releases/tiny-llm-7b-release1/release_info.json`
- `models/releases/tiny-llm-7b-release1/merged_model/*.gguf`
- LM Studio model folder under `%USERPROFILE%/.lmstudio/models/<publisher>/tiny-llm-7b-release1`

## 7) LM Studio Settings

- Prompt Template: model default (`Auto`), not `Empty`
- Temperature for compliance tests: `0.0` to `0.2`

## 8) Quick Smoke Test Prompts

1. `Write a Python function is_prime(n). Return only a fenced python block.`
2. `Return ONLY valid JSON {"language": string, "has_code": boolean, "code": string}. Task: write add(a,b).`
3. `Give exactly 3 bullet points on risks of fine-tuning with small datasets.`
4. `Compute 47*19. Reply with: "<number>. <one short sentence>".`

## Troubleshooting Shortlist

- `Out of memory`: lower `--max_length`, reduce batch size, or switch from 4-bit QLoRA to LoRA fallback profile.
- `No checkpoints found`: verify dataset globs and `--min_loaded_examples` constraints.
- `No model in API registry`: ensure release path exists and matches `tiny-llm-7b-release1` naming.
- `Slow runtime`: prefer CUDA profile or use root quickstart with 0.5B runtime model.
