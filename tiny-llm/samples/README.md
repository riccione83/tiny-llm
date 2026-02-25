# Local Sample Data

This folder ships with small, high-quality local samples for quick training runs.

- `base/`: short knowledge-oriented text for base training experiments (continued pretraining or scratch initialization bootstrap).
- `sft/`: instruction-style JSONL for LoRA alignment.

These samples are intentionally small. They are useful for smoke tests and
for bootstrapping behavior, but they are not a substitute for large-scale
training data.
