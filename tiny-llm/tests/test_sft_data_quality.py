import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sft_data_quality import (  # noqa: E402
    SftHygieneConfig,
    apply_sft_hygiene,
    looks_like_python_code,
    parse_jsonl_object_line,
    row_has_no_markdown_instruction,
)


class DataQualityTests(unittest.TestCase):
    def test_strict_jsonl_valid_object(self):
        row, err = parse_jsonl_object_line('{"messages":[{"role":"user","content":"hi"}]}')
        self.assertIsNone(err)
        self.assertIsInstance(row, dict)

    def test_strict_jsonl_multiple_objects_on_one_line(self):
        row, err = parse_jsonl_object_line('{"a":1} {"b":2}')
        self.assertIsNone(row)
        self.assertIsNotNone(err)
        self.assertIn("multiple JSON objects", err)

    def test_strict_jsonl_unterminated_string_has_newline_hint(self):
        row, err = parse_jsonl_object_line('{"a":"hello')
        self.assertIsNone(row)
        self.assertIsNotNone(err)
        self.assertIn("possible unescaped literal newline", err)

    def test_python_hygiene_normalize_wraps_fence(self):
        row = {
            "messages": [
                {"role": "user", "content": "Write a Python function"},
                {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
            ]
        }
        cfg = SftHygieneConfig(code_fence_mode="normalize")
        res = apply_sft_hygiene(row=row, answer="def add(a, b):\n    return a + b", cfg=cfg)
        self.assertTrue(res.keep)
        self.assertIn("```python", res.answer.lower())
        self.assertIn("def add", res.answer)

    def test_python_hygiene_rejects_conflicting_no_markdown(self):
        row = {
            "messages": [
                {"role": "system", "content": "No markdown, code only."},
                {"role": "user", "content": "Write Python code."},
            ]
        }
        cfg = SftHygieneConfig(code_fence_mode="reject", reject_no_markdown_code_instructions=True)
        res = apply_sft_hygiene(row=row, answer="def f(x):\n    return x", cfg=cfg)
        self.assertFalse(res.keep)
        self.assertIn("no_markdown", res.reason)

    def test_python_detection(self):
        self.assertTrue(looks_like_python_code("def f(x):\n    return x"))
        self.assertFalse(looks_like_python_code("The capital of Italy is Rome."))

    def test_detect_no_markdown_instruction(self):
        row = {"messages": [{"role": "system", "content": "Please output only text, no markdown."}]}
        self.assertTrue(row_has_no_markdown_instruction(row))


if __name__ == "__main__":
    unittest.main()
