import math
import re
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


def _tokset(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class ChunkRetriever:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedding_model_name = embedding_model_name
        self._embedder = None
        self._emb_matrix: Optional[np.ndarray] = None
        self._chunks: List[Tuple[str, str]] = []
        self._lexical_only = False

    def _lazy_load_embedder(self) -> None:
        if self._embedder is not None or self._lexical_only:
            return
        try:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            from transformers.utils import logging as hf_logging  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            hf_logging.set_verbosity_error()
            self._embedder = SentenceTransformer(self.embedding_model_name)
        except Exception:
            self._lexical_only = True
            self._embedder = None

    def build_index(self, chunks_with_source: List[Tuple[str, str]]) -> None:
        self._chunks = [(t, s) for (t, s) in chunks_with_source if (t or "").strip()]
        if not self._chunks:
            self._emb_matrix = None
            return
        self._lazy_load_embedder()
        if self._lexical_only:
            self._emb_matrix = None
            return
        texts = [t for (t, _) in self._chunks]
        emb = self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._emb_matrix = np.asarray(emb, dtype=np.float32)

    def _retrieve_lexical(self, question: str, k: int) -> List[RetrievedChunk]:
        q_toks = _tokset(question)
        scored: List[RetrievedChunk] = []
        for text, source in self._chunks:
            toks = _tokset(text)
            if not toks:
                continue
            overlap = len(q_toks & toks)
            # Mild normalization by token count.
            score = overlap / math.sqrt(max(1, len(toks)))
            if score > 0:
                scored.append(RetrievedChunk(text=text, source=source, score=float(score)))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(1, int(k))]

    def retrieve(self, question: str, k: int = 5) -> List[RetrievedChunk]:
        if not self._chunks:
            return []
        if self._emb_matrix is None:
            return self._retrieve_lexical(question, k=k)

        q_emb = self._embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q_emb, dtype=np.float32)[0]
        sims = self._emb_matrix @ q
        idx = np.argsort(-sims)[: max(1, int(k))]
        out: List[RetrievedChunk] = []
        for i in idx.tolist():
            text, source = self._chunks[int(i)]
            out.append(RetrievedChunk(text=text, source=source, score=float(sims[int(i)])))
        return out
