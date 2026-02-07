from dataclasses import dataclass


@dataclass
class AssistantConfig:
    backend: str = "hf"  # "hf" | "tiny"
    llm_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    tiny_ckpt: str = "checkpoints_v2/final.pt"
    tiny_tokenizer: str = "tokenizer.model"
    tiny_lora: str = ""
    tiny_top_p: float = 1.0
    max_new_tokens: int = 160
    temperature: float = 0.0
    top_k: int = 5
    search_results: int = 5
    chunk_chars: int = 1100
    chunk_overlap: int = 180
    timeout_sec: int = 20
    max_context_chars: int = 6000
    direct_confidence_threshold: float = 0.72
    direct_max_sentences: int = 2
