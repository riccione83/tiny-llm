# Tiny LLM (From Scratch + LoRA SFT)

Train and iterate a small GPT-style model from random initialization, then adapt it with LoRA for:

- chat-style instruction following
- short constrained answers (number-only, YES/NO-only)
- user-pasted summarization
- optional web-assisted responses in chat runtime

This repository is optimized for practical iteration on a local machine (Windows + CUDA).

## Current Model Snapshot

- Base architecture: decoder-only GPT (RoPE + RMSNorm + SwiGLU)
- Size: ~190M parameters (`09_chat.py` prints `190,182,272`)
- Tokenizer: SentencePiece (`tokenizer.model`, vocab 32k)
- SFT method: LoRA on attention projections (`attn.qkv`, `attn.proj`)
- Trainable LoRA params: ~1.38M

## What The Model Can Do

- Basic conversational replies
- Follow explicit constraints when examples are present (e.g., number-only / YES-NO)
- Summarize pasted text in 2 sentences
- Summarize noisy “web-like” pasted snippets
- For web queries, runtime can fetch/search and build an answer with source links

## What It Cannot Reliably Do (Yet)

- Guaranteed factual correctness without strong/clean sources
- Deep reasoning over complex multi-hop questions
- Consistent “latest” answers across all domains with perfect source quality
- Production-grade citation quality and source verification

In short: good for iterative prototyping and controlled workflows; not yet a fully reliable general assistant.

## Repository Layout

- `00_build_pretrain_corpus.py` - stream/build pretrain text corpus
- `01_train_tokenizer.py` - train SentencePiece tokenizer
- `02_pretokenize.py` - tokenize corpus into `.npy`
- `03_train_base_v2.py` - base pretraining
- `04_make_chat_sft.py` - chat SFT data
- `05_make_routing_sft.py` - constraints/routing SFT data
- `06_make_summarize_sft.py` - summarization SFT data
- `07_merge_sft_jsonl.py` - dataset mixing/ratio control
- `08_train_summarize_lora.py` - LoRA SFT training
- `09_chat.py` - interactive chat runtime (model + web fallback + logs)
- `10_make_hardcases_sft.py` - hard-case generation for micro-retrain
- `11_make_daily_micro_sft.py` - convert logs to daily SFT mix
- `12_daily_micro_retrain.ps1` - one-shot daily micro-retrain script

Data/artifacts are under:

- `data/`
- `checkpoints_v2/`
- `finetuning_v2/`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## End-to-End Training Pipeline

### 0) Build pretraining corpus

```powershell
python 00_build_pretrain_corpus.py --out data/pretrain.txt --target_gb 5 --dataset wikipedia --config 20220301.en --split train --text_key text --min_chars 500 --min_ascii 0.92 --min_letters 0.55
```

### 1) Train tokenizer

```powershell
python 01_train_tokenizer.py --input data/pretrain.txt --vocab_size 32000 --model_prefix tokenizer
```

### 2) Pretokenize

```powershell
python 02_pretokenize.py --input data/pretrain.txt --tokenizer tokenizer.model --out data/pretrain_tokens.npy
```

### 3) Base pretrain

```powershell
python 03_train_base_v2.py --tokens data/pretrain_tokens.npy --tokenizer tokenizer.model --out_dir checkpoints_v2 --steps 200000 --block_size 768 --batch_size 16 --grad_accum 8 --lr 8e-5 --warmup 500 --save_every 500 --print_every 50 --sample_every 2000 --resume
```

### 4) Build SFT datasets

```powershell
python 04_make_chat_sft.py --out data/basic_chat_sft.jsonl --seed 42 --n 120000
python 05_make_routing_sft.py --out data/routing_constraints_sft.jsonl --seed 42 --n 30000
python 06_make_summarize_sft.py --out data/summarize_sft.jsonl --tokenizer tokenizer.model --max 600000 --seed 42 --splits train validation test --block_size 768 --max_tokens 760 --truncate_input --web_noise_prob 0.1
```

### 5) Merge SFT mix

```powershell
python 07_merge_sft_jsonl.py --summarize data/summarize_sft.jsonl --chat data/basic_chat_sft.jsonl --routing data/routing_constraints_sft.jsonl --out data/sft_merged.jsonl --max_rows 400000 --summarize_ratio 0.70 --routing_ratio 0.10 --seed 42
```

### 6) Train LoRA

```powershell
python 08_train_summarize_lora.py --base_ckpt checkpoints_v2/final.pt --tokenizer tokenizer.model --sft_jsonl data/sft_merged.jsonl --out_dir finetuning_v2 --epochs 2 --batch_size 16 --grad_accum 8 --lr 1.5e-4 --warmup 200 --print_every 20 --sample_every 200 --save_every 500
```

### 7) Run chat

```powershell
python 09_chat.py --base_ckpt checkpoints_v2/final.pt --tokenizer tokenizer.model --lora_adapter finetuning_v2/lora_adapter.pt --temperature 0.0 --top_p 1.0 --confidence_threshold 0.18 --web_results 3
```

## Daily Improvement Loop (Optional)

Chat runtime logs go to:

- `data/chat_turns_log.jsonl`
- `data/web_chat_log.jsonl`

Build daily dataset + micro-retrain:

```powershell
powershell -ExecutionPolicy Bypass -File 12_daily_micro_retrain.ps1
```

Dataset only (no training):

```powershell
powershell -ExecutionPolicy Bypass -File 12_daily_micro_retrain.ps1 -NoTrain
```

## Notes / Known Risks

- From-scratch pretraining quality is token-budget dependent.
- Web answers depend on source quality and ranking heuristics in runtime.
- Daily self-training must keep anchor data in mix to avoid regressions.
- Always run fixed prompt checks before replacing your current LoRA adapter.
