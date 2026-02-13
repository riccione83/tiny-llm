import unittest
from pathlib import Path


class ReleaseScriptContractTests(unittest.TestCase):
    def test_release_script_invokes_merge_and_gguf_tools(self):
        text = Path("tiny-llm/release_lmstudio.ps1").read_text(encoding="utf-8")
        self.assertIn("06_merge_lora_checkpoint.py", text)
        self.assertIn("07_verify_gguf_chat_template.py", text)
        self.assertIn("convert_hf_to_gguf.py", text)
        self.assertIn("llama-quantize.exe", text)

    def test_release_script_exposes_cleanup_flags(self):
        text = Path("tiny-llm/release_lmstudio.ps1").read_text(encoding="utf-8")
        self.assertIn("$CleanupOldCheckpoints", text)
        self.assertIn("$CleanupOldLmStudioModels", text)

    def test_merge_helper_has_required_args(self):
        text = Path("tiny-llm/06_merge_lora_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn("--base_model_dir", text)
        self.assertIn("--adapter_checkpoint", text)
        self.assertIn("--output_dir", text)


if __name__ == "__main__":
    unittest.main()
