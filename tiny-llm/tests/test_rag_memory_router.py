import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path("tiny-llm/rag_memory_router.py").resolve()
SPEC = importlib.util.spec_from_file_location("rag_memory_router", MODULE_PATH)
RAG = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = RAG
SPEC.loader.exec_module(RAG)


class RagMemoryRouterTests(unittest.TestCase):
    def test_route_auto_falls_back_to_local_without_cloud_key(self):
        route, reason = RAG.route_query(
            query="Compare architecture tradeoffs for this system.",
            mode="auto",
            has_cloud_api_key=False,
        )
        self.assertEqual(route, "local")
        self.assertIn("missing", reason.lower())

    def test_route_auto_selects_cloud_for_deep_prompt(self):
        route, reason = RAG.route_query(
            query="Compare architecture tradeoffs and propose a long-term strategy.",
            mode="auto",
            has_cloud_api_key=True,
        )
        self.assertEqual(route, "cloud")
        self.assertIn("deeper", reason.lower())

    def test_retrieve_context_returns_relevant_chunk(self):
        chunk = RAG.KnowledgeChunk(
            source="doc.txt",
            text="LoRA adapters reduce trainable parameters for fine-tuning.",
            tokens=RAG.tokenize_set("LoRA adapters reduce trainable parameters for fine-tuning."),
        )
        rows = RAG.retrieve_context(
            query="What do LoRA adapters reduce during fine-tuning?",
            chunks=[chunk],
            top_k=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0][1], 0.0)

    def test_memory_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            mem = Path(td) / "session.jsonl"
            RAG.append_memory_turn(mem, user="u1", assistant="a1", route="local")
            RAG.append_memory_turn(mem, user="u2", assistant="a2", route="cloud")
            msgs = RAG.load_memory_messages(mem, max_turns=1)
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]["role"], "user")
            self.assertEqual(msgs[0]["content"], "u2")
            self.assertEqual(msgs[1]["role"], "assistant")
            self.assertEqual(msgs[1]["content"], "a2")


if __name__ == "__main__":
    unittest.main()
