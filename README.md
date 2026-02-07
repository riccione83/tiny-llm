# Mini Assistant (Grounded Web QA)

This repo supports two official paths:

- `mini_assistant/` (recommended): grounded QA with web retrieval.
- `tiny-llm/`: full local end-to-end training pipeline.

Use `mini_assistant/` if you want immediate quality and simpler setup.
Use `tiny-llm/` if you want to train everything locally from scratch.

## Path A (Recommended): Ready Model + Grounded QA

- LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Retriever embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Runtime package: `mini_assistant/`
- Optional local backend: your custom tiny checkpoint from `09_chat.py`

## Quick Start

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Run chat (search web automatically):

```powershell
python -m mini_assistant.chat --backend hf --model_name Qwen/Qwen2.5-1.5B-Instruct --embedding_model sentence-transformers/all-MiniLM-L6-v2 --temperature 0.0
```

Show routing/debug (`direct` vs `web`):

```powershell
python -m mini_assistant.chat --show_debug --direct_confidence_threshold 0.72
```

Run chat on a fixed URL:

```powershell
python -m mini_assistant.chat --backend hf --url https://en.wikipedia.org/wiki/Italy
```

Run chat with your ready local tiny checkpoint:

```powershell
python -m mini_assistant.chat --backend tiny --tiny_ckpt checkpoints_v2/final.pt --tiny_tokenizer tokenizer.model --tiny_lora finetuning_v2/lora_adapter.pt --tiny_top_p 1.0 --temperature 0.0 --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

## Path B: Local End-to-End Training (No External Base LLM)

Training scripts are in `tiny-llm/`.

```powershell
cd tiny-llm
python .\00_start.py
```

Or run the explicit sequence:

```powershell
cd tiny-llm
python .\01_make_chat_corpus_and_tokenize.py
python .\02_train_base_chat.py
python .\03_create_instruct.py
python .\05_make_synth_chat_sft.py
python .\06_make_feedback_sft.py
python .\07_lora_and_chat.py --mode synth_lora
python .\07_lora_and_chat.py --mode feedback_lora
python .\07_lora_and_chat.py --mode chat --use_lora
```

Full details: `tiny-llm/README.md`.

Run regression eval:

```powershell
python 13_eval_chat_quality.py --backend hf --model_name Qwen/Qwen2.5-1.5B-Instruct --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

Run confidence-gate check:

```powershell
python -m mini_assistant.eval_confidence_gate --backend hf --model_name Qwen/Qwen2.5-1.5B-Instruct --embedding_model sentence-transformers/all-MiniLM-L6-v2 --direct_confidence_threshold 0.72
python -m mini_assistant.eval_confidence_gate --backend tiny --tiny_ckpt checkpoints_v2/final.pt --tiny_tokenizer tokenizer.model --tiny_lora finetuning_v2/lora_adapter.pt --embedding_model sentence-transformers/all-MiniLM-L6-v2 --direct_confidence_threshold 0.72
```

## Notes

- The new entrypoint for practical usage is `mini_assistant/chat.py`.
- Best quality today is with `--backend hf` (`Qwen/Qwen2.5-1.5B-Instruct`).
- `--backend tiny` works with your local checkpoint, but routing quality is lower than `hf`.

## What To Ask

Direct answers (no URL needed):

- `Hi`
- `What is the capital of Italy?`
- `Reply YES or NO only: Is Berlin in Germany?`
- `Quanto fa 144/12? Rispondi solo con un numero.`

Web-grounded answers:

- `What is the VRAM of NVIDIA RTX 5070 Ti? Use official sources only.`
- `What is the latest NVIDIA GPU line?`
- `Summarize the key points of this page in 3 bullets.` with `/url <page>`

URL-focused mode:

1. `/url https://en.wikipedia.org/wiki/Italy`
2. Ask: `What is the official language?`
