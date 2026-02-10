#!/usr/bin/env python3
"""
Compatibility shim (old module path).

Use: `python -m mini_assistant.eval_confidence_gate`
New location: `python -m mini_assistant.evals.confidence_gate`
"""

from .evals.confidence_gate import main


if __name__ == "__main__":
    main()

