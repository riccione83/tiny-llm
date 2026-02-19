# tiny_LLM

[![CI](https://github.com/riccione83/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/riccione83/tiny-llm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready local LLM workspace with two official paths:

- `mini_assistant/` (recommended): grounded Q&A with web retrieval and a simple CLI.
- `tiny-llm/`: full training and release pipeline (LoRA/QLoRA, evaluation, LM Studio export).

The repository is organized for repeatability and public readability: quick onboarding at root, detailed runbooks in `tiny-llm/`.

## Repository Layout

```text
.
|-- mini_assistant/         # Runtime assistant package (chat, routing, retrieval, evals)
|-- tiny-llm/               # Training/eval/release pipeline and datasets samples
|-- model_api_server.py     # OpenAI-compatible local API server
|-- eval.py                 # Unified evaluation entrypoint (grounded/chat/both)
|-- scripts/                # Operational helper scripts
|-- requirements.txt
`-- README.md
```

## Quick Start (Recommended Path)

### 1) Install dependencies

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 2) Start the grounded assistant

```powershell
python -m mini_assistant.chat --backend hf --model_name Qwen/Qwen3-4B-Instruct-2507 --embedding_model sentence-transformers/all-MiniLM-L6-v2 --temperature 0.0
```

### 3) Optional: start local OpenAI-compatible API

```powershell
python .\model_api_server.py --host 127.0.0.1 --port 8001 --default_model tiny-llm-7b
```

Quick smoke test:

```powershell
.\scripts\api_smoke_test.ps1
```

## Evaluation

Run the compatibility wrapper:

```powershell
python .\eval.py --suite grounded --backend hf --model_name Qwen/Qwen3-4B-Instruct-2507 --embedding_model sentence-transformers/all-MiniLM-L6-v2
python .\eval.py --suite chat --backend hf --model_name Qwen/Qwen3-4B-Instruct-2507
```

Run unit tests:

```powershell
python -m unittest discover -s tiny-llm/tests -p "test_*.py"
```

Or use helper script:

```powershell
.\scripts\check.ps1
```

## Training and Release (Advanced)

The official 7B runbook lives in `tiny-llm/README.md`.

That path includes:
- base model download
- LoRA/QLoRA training
- checkpoint ranking
- regression suite
- LM Studio release packaging

## Legacy Compatibility

- `legacy_chat.py` is retained for legacy tiny-checkpoint flows.
- `mini_assistant --backend tiny` depends on legacy artifacts and is optional.

## Contributing

Contribution workflow and quality expectations are documented in `CONTRIBUTING.md`.

## License

This project is licensed under the MIT License. See `LICENSE`.
