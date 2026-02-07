# Tiny LLM Chat Pipeline

Small GPT-style model (~184M params) focused on chat-first behavior on a single GPU (Windows).

This folder is the local end-to-end training path for this repository.

## Quick Start

Run the menu:

```
python .\00_start.py
```

## Main Steps (v1)

1) Build chat corpus + tokenize

```
python .\01_make_chat_corpus_and_tokenize.py
```

Outputs:
- data\chat_corpus_v1.txt
- data\chat_corpus_v1_tokens.npy

2) Train base chat model

```
python .\02_train_base_chat.py
```

Outputs:
- checkpoints_chat_v1\*.pt
- training_chat_v1.csv

3) Build instruction dataset (JSON)

```
python .\03_create_instruct.py
```

Output:
- data\instruct_v4.json

4) Small synthetic chat set

```
python .\04_make_synthetic_chat.py
```

5) Large synthetic SFT set

```
python .\05_make_synth_chat_sft.py
```

6) Build feedback dataset (auto-generated)

```
python .\06_make_feedback_sft.py
```

7) LoRA fine-tune + chat

```
python .\07_lora_and_chat.py --mode lora
python .\07_lora_and_chat.py --mode chat --use_lora
```

Outputs:
- finetuning\lora_adapter.pt
- finetuning\lora_full_state.pt

## Recommended End-to-End Sequence (v1)

Use this order from scratch:

1) Build chat corpus + tokenize

```
python .\01_make_chat_corpus_and_tokenize.py
```

2) Train base chat model

```
python .\02_train_base_chat.py
```

3) Build instruction dataset

```
python .\03_create_instruct.py
```

4) Build small synthetic chat set (optional)

```
python .\04_make_synthetic_chat.py
```

5) Build large synthetic SFT set

```
python .\05_make_synth_chat_sft.py
```

6) Build feedback dataset (auto-generated)

```
python .\06_make_feedback_sft.py
```

7) LoRA on synthetic SFT set

```
python .\07_lora_and_chat.py --mode synth_lora
```

8) LoRA on feedback data

```
python .\07_lora_and_chat.py --mode feedback_lora
```

9) Chat test

```
python .\07_lora_and_chat.py --mode chat --use_lora
```

## RAG (Web Search) in Chat

Chat mode now uses web search + embeddings by default. To disable:

```
python .\07_lora_and_chat.py --mode chat --use_lora --rag_mode off
```

Helpful flags:

- `--rag_mode auto|always|off` (default: auto)
- `--rag_site wikipedia.org` (restrict to a site)
- `--rag_debug` (print retrieved context)
- `--rag_top_k 6` and `--rag_web_k 8` (depth)
- `--rag_no_extract` (use model instead of extractive answer)

Install dependencies:

```
pip install -U ddgs sentence-transformers requests beautifulsoup4
```

RAG cache is stored in `rag_cache/`.

## Synthetic Data Utilities

Small chat set:

```
python .\04_make_synthetic_chat.py
```

Large SFT set (JSONL with instruction/output):

```
python .\05_make_synth_chat_sft.py
```

Large feedback set (JSONL with instruction/chosen):

```
python .\06_make_feedback_sft.py
```

## v2 Scripts (optional)

Create larger instruction data:

```
python .\10_create_instruct_v2.py
```

Train a v2 base model:

```
python .\11_train_base_v2.py
```

## Notes

- All scripts are resumable where possible.
- LoRA uses the latest base checkpoint (prefers final.pt if present).
- Large feedback and synth sets use streaming datasets to avoid RAM issues.

## File Map

- 00_start.py                Menu runner
- 01_make_chat_corpus_and_tokenize.py
- 02_train_base_chat.py
- 03_create_instruct.py
- 04_make_synthetic_chat.py
- 05_make_synth_chat_sft.py
- 06_make_feedback_sft.py
- 07_lora_and_chat.py
- 08_auto_feedback_1k.py
- 09_clean_dataset.py
- 10_create_instruct_v2.py
- 11_train_base_v2.py

