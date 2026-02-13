import re
import unittest
from pathlib import Path


class TrainCliContractTests(unittest.TestCase):
    def test_local_jsonl_glob_uses_append(self):
        text = Path("tiny-llm/04_train_lora.py").read_text(encoding="utf-8")
        m = re.search(
            r'--local_jsonl_glob".*?action="append"',
            text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(m, "--local_jsonl_glob must remain argparse append mode")

    def test_targeted_repair_script_passes_multiple_local_globs(self):
        text = Path("tiny-llm/run_lora_targeted_repair.ps1").read_text(encoding="utf-8")
        hits = re.findall(r'--local_jsonl_glob\s+"[^"]+"', text)
        self.assertGreaterEqual(
            len(hits),
            2,
            "run_lora_targeted_repair.ps1 should pass multiple --local_jsonl_glob values",
        )


if __name__ == "__main__":
    unittest.main()
