# tiny-llm (Modern Minimal Pipeline)

This folder now uses a strict 4-script workflow:

1. `01_download_base.py`
2. `02_train_base.py`
3. `03_download_lora_base.py`
4. `04_train_lora.py`

Goal: a modern, maintainable pipeline that can run for days and scale quality with bigger data.

Included local samples:
- Base training: `samples/base/*.txt` and `samples/base/*.jsonl`
- LoRA training: `samples/sft/*.jsonl`

These are already loaded by default by `02_train_base.py` and `04_train_lora.py`.

## Install

```powershell
python -m pip install -U pip
python -m pip install -r ..\requirements.txt
```

## 1) Download Base Model (full-training path)

```powershell
cd tiny-llm
python .\01_download_base.py --model_id Qwen/Qwen2.5-0.5B-Instruct --output_dir models/base
```

Notes:
- `0.5B` is a practical default for full fine-tuning.
- Change `--model_id` if you have more VRAM.

## 2) Train Base Model (knowledge-first)

```powershell
python .\02_train_base.py --model_dir models/base --output_dir models/base_trained --recipe knowledge-heavy --max_steps 30000 --repeat_sources --gradient_checkpointing
```

What this script does:
- Streams large corpora (FineWeb-EDU + C4 + FineWeb, recipe dependent).
- Packs tokens into fixed blocks for efficient causal LM training.
- Supports long multi-day runs using `--max_steps`.
- Saves periodic checkpoints every `--save_steps`.
- On `Ctrl+C`, saves an emergency checkpoint before exit.
- Includes local curated samples by default.

Useful options:
- `--recipe tiny|standard|knowledge-heavy`
- `--max_steps 100000` for long training
- `--hf_source "dataset|config|split|text_field|max_texts"` to add more sources
- `--disable_hf_data` to train only on local samples/custom files
- `--resume_from_checkpoint <path>`

Resume example:

```powershell
python .\02_train_base.py --model_dir models/base --output_dir models/base_trained --resume_from_checkpoint models/base_trained/checkpoint-1500
```

## 3) Download LoRA Base (alignment path)

```powershell
python .\03_download_lora_base.py --model_id Qwen/Qwen3-4B-Instruct-2507 --output_dir models/lora_base
```

Why separate:
- For best chat quality, LoRA often benefits from a stronger instruct base than full-training default.

## 4) Train LoRA Adapter (instruction/chat)

```powershell
python .\04_train_lora.py --model_dir models/lora_base --output_dir models/lora_adapter --recipe heavy --max_steps 8000 --repeat_sources --gradient_checkpointing --save_merged
```

What this script does:
- Trains PEFT LoRA on instruction/chat datasets (UltraChat, OpenOrca, Alpaca, Dolly by recipe).
- Uses completion-only labels (prompt tokens masked with `-100`).
- Saves adapter and optional merged model.
- Saves periodic checkpoints every `--save_steps`.
- On `Ctrl+C`, saves an emergency checkpoint before exit.
- Includes local curated SFT samples by default.

Useful options:
- `--target_modules auto` (default) or manual comma list
- `--lora_r`, `--lora_alpha`, `--lora_dropout`
- `--hf_source "dataset|config|split|max_rows"` to add more SFT data
- `--disable_hf_data` to train only on local SFT JSONL
- `--resume_from_checkpoint <path>`

Resume example:

```powershell
python .\04_train_lora.py --model_dir models/lora_base --output_dir models/lora_adapter --resume_from_checkpoint models/lora_adapter/checkpoint-900
```

## Quick Smoke Runs (with built-in samples)

```powershell
python .\02_train_base.py --model_dir models/base --output_dir models/base_trained --disable_hf_data --max_steps 300 --save_steps 100 --gradient_checkpointing
python .\04_train_lora.py --model_dir models/lora_base --output_dir models/lora_adapter --disable_hf_data --max_steps 400 --save_steps 100 --gradient_checkpointing
```

## Quality Guidance

- For stronger knowledge:
  - prioritize `02_train_base.py` with `--recipe knowledge-heavy`
  - run higher `--max_steps` (tens of thousands to hundreds of thousands)
- For better instruction behavior:
  - follow with `04_train_lora.py` on `--recipe standard` or `heavy`
- For best results, monitor loss and periodically evaluate with a fixed benchmark set.

## Output Layout

- `models/base`              downloaded base for full training
- `models/base_trained`      continued-pretrained model
- `models/lora_base`         downloaded base for LoRA
- `models/lora_adapter`      LoRA adapter (+ optional merged model)
