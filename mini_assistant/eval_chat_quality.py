#!/usr/bin/env python3
"""
Compatibility shim (old module path).

Use: `python -m mini_assistant.eval_chat_quality`
New location: `python -m mini_assistant.evals.chat_quality`
"""

from .evals.chat_quality import main


if __name__ == "__main__":
    main()

