# tiny_LLM

[![CI](https://github.com/riccione83/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/riccione83/tiny-llm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready local LLM workspace with two official paths:

- `mini_assistant/` (recommended): grounded Q&A with web retrieval and a simple CLI.
- `tiny-llm/`: full training and release pipeline (LoRA/QLoRA, evaluation, LM Studio export).

If you want base training (continued pretraining or scratch initialization), use:
[`tiny-llm/02_train_base.py`](tiny-llm/02_train_base.py)

## Before You Start

- Python `3.10+`
- `pip` available in shell
- Internet access for first model download and web retrieval
- Optional GPU for speed (CUDA strongly recommended for training)

## Platform Support Matrix

| Platform Profile | Runtime Assistant | Evaluations | LoRA Training | QLoRA 4-bit | LM Studio Release Packaging |
|---|---|---|---|---|---|
| Windows + NVIDIA CUDA | Yes | Yes | Yes | Yes | Yes |
| Linux + NVIDIA CUDA | Yes | Yes | Yes | Yes | No (use Windows for release packaging) |
| macOS Apple Silicon (MPS/Metal) | Yes | Yes | Yes (LoRA) | No | No |
| CPU-only (any OS) | Yes (slow) | Yes (slow) | Limited/slow | No | No |

Notes:
- Training/SFT Python flows are cross-platform.
- `tiny-llm/scripts/release_lmstudio.ps1` is intentionally Windows-only.

## Expected Speed (Indicative)

| Profile | First Model Download + Cache Warmup | Chat Latency (0.5B) | 7B LoRA/QLoRA (300 steps) | Practical Recommendation |
|---|---|---|---|---|
| CUDA 16GB+ VRAM | 5-20 min | 1-6 s/answer | 1-4 h | Best end-to-end experience |
| Apple Silicon 16-32GB unified memory | 5-20 min | 2-10 s/answer | 3-10 h (LoRA only) | Good for learning and iteration |
| CPU-only | 5-20 min | 10-60+ s/answer | Often impractical | Use runtime/eval path first |

Important: `5-20 min` refers to initial download and local cache warmup (runtime model + embedding model), not training.
All timings are indicative and depend on network, storage, thermals, and model cache state.

## Base Training with Scratch Initialization (50GB Corpus, Indicative)

This is a planning table for full base training workloads on a `~50 GB` corpus.
Baseline reference: `RTX 5070 Ti 16GB` is estimated at about `5 days` running `24/7`.

| System Profile | Relative Throughput vs RTX 5070 Ti 16GB | Estimated Wall-Clock for 50GB (24/7) |
|---|---|---|
| RTX 4090 24GB | 1.4x-2.0x | 2.5-3.5 days |
| RTX 5080/4080 class 16GB | 0.9x-1.2x | 4-6 days |
| RTX 5070 Ti 16GB | 1.0x | about 5 days |
| RTX 4070 Ti/SUPER 16GB | 0.6x-0.85x | 6-9 days |
| Apple Silicon Max (M2/M3 class) | 0.35x-0.55x | 9-14 days |
| CPU-only (desktop/server) | 0.08x-0.20x | 25-60 days |

For assumptions and notes, see the full 7B runbook section:
[`tiny-llm/README.md`](tiny-llm/README.md)

`02_train_base.py` now supports both base initialization modes:

```bash
# Continued pretraining (load existing weights from --model_dir)
python tiny-llm/02_train_base.py --init_mode pretrained --model_dir models/base --output_dir models/base_trained

# Scratch initialization (random weights from config + tokenizer)
python tiny-llm/02_train_base.py --init_mode scratch --config_source Qwen/Qwen2.5-0.5B-Instruct --output_dir models/base_scratch
```

Scratch-initialization quickstart action IDs:
- `tiny.train.base.05b.scratch`
- `tiny.train.base.3b.scratch`
- `tiny.train.base.3b.scratch.wiki`
- `tiny.train.base.7b.scratch`

## Repository Layout

```text
.
|-- mini_assistant/         # Runtime assistant package (chat, routing, retrieval, evals)
|-- tiny-llm/               # Training/eval/release pipeline and dataset samples
|-- model_api_server.py     # OpenAI-compatible local API server
|-- eval.py                 # Unified evaluation entrypoint (grounded/chat/both)
|-- scripts/                # Operational helper scripts
|-- requirements.txt
`-- README.md
```

## Quick Start (Recommended Path)

### 0) Quickstart launcher (menu)

```bash
python ./quickstart.py
```

Useful non-interactive variants:

```bash
python ./quickstart.py --list-actions
python ./quickstart.py --run env.install --yes
python ./quickstart.py --run mini.chat --yes
python ./quickstart.py --run api.server.default --yes
python ./quickstart.py --no-anim
```

### 1) Install dependencies (manual path)

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Expected result: installation completes without errors.

### 2) Profile quickstarts (copy/paste)

#### Windows + NVIDIA CUDA (PowerShell)

```powershell
python .\quickstart.py --run env.install --yes
python .\quickstart.py --run mini.chat --yes
```

#### Linux + NVIDIA CUDA (bash)

```bash
python ./quickstart.py --run env.install --yes
python ./quickstart.py --run mini.chat --yes
```

#### macOS Apple Silicon (Metal/MPS)

```bash
python ./quickstart.py --run env.install --yes
python -m mini_assistant.chat --backend hf --model_name Qwen/Qwen2.5-0.5B-Instruct --embedding_model sentence-transformers/all-MiniLM-L6-v2 --temperature 0.0
```

Training note for macOS: use LoRA mode, not 4-bit QLoRA (`--use_4bit` off in the 7B runbook).

#### CPU-only (any OS)

```bash
python ./quickstart.py --run env.install --yes
python -m mini_assistant.chat --backend hf --model_name Qwen/Qwen2.5-0.5B-Instruct --embedding_model sentence-transformers/all-MiniLM-L6-v2 --temperature 0.0 --max_new_tokens 96
```

CPU-only note: this mode is for learning/validation; response latency can be high.

### 3) Optional: OpenAI-compatible local API

```bash
python ./model_api_server.py --host 127.0.0.1 --port 8001 --default_model base-qwen-0.5b
```

PowerShell smoke test:

```powershell
./scripts/api_smoke_test.ps1
```

Force test on 7B release model (if already exported):

```powershell
./scripts/api_smoke_test.ps1 -Models tiny-llm-7b
```

Bash/curl smoke test (Linux/macOS, no PowerShell required):

```bash
curl -s http://127.0.0.1:8001/v1/models
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"base-qwen-0.5b","messages":[{"role":"user","content":"Say hello in one short sentence."}],"temperature":0.0,"max_tokens":64}'
```

## Evaluation

Run compatibility wrapper:

```bash
python ./eval.py --suite grounded --backend hf --model_name Qwen/Qwen2.5-0.5B-Instruct --embedding_model sentence-transformers/all-MiniLM-L6-v2
python ./eval.py --suite chat --backend hf --model_name Qwen/Qwen2.5-0.5B-Instruct
```

Run unit tests:

```bash
python -m unittest discover -s tiny-llm/tests -p "test_*.py"
```

Optional helper script:

```powershell
./scripts/check.ps1
```

PowerShell Core variant on Linux/macOS:

```bash
pwsh ./scripts/check.ps1
```

## Training and Release (Advanced)

Official 7B runbook: [`tiny-llm/README.md`](tiny-llm/README.md)

Includes:
- base model download
- LoRA/QLoRA training
- base training time-budget table for 50GB corpora
- checkpoint ranking
- regression suite
- LM Studio release packaging

## Legacy Compatibility

- `legacy_chat.py` is retained for legacy tiny-checkpoint flows.
- `mini_assistant --backend tiny` depends on legacy artifacts and is optional.

## Contributing

Contribution workflow and quality expectations: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
