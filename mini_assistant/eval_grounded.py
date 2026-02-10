#!/usr/bin/env python3
"""
Compatibility shim (old module path).

Use: `python -m mini_assistant.eval_grounded`
New location: `python -m mini_assistant.evals.grounded`
"""

from .evals.grounded import main


if __name__ == "__main__":
    main()

