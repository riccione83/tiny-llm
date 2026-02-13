import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regression_suite import MockGenerator, default_cases, run_suite  # noqa: E402


class RegressionSuiteTests(unittest.TestCase):
    def test_mock_backend_passes_required_cases(self):
        passed, total, failures = run_suite(MockGenerator(), default_cases())
        self.assertEqual(total, 4)
        self.assertEqual(passed, total)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
