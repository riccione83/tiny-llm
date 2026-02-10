#!/usr/bin/env python3
"""
Compatibility shim (old filename).

The legacy tiny model chat CLI was moved to `legacy_chat.py`.
This wrapper keeps:
- `python 09_chat.py ...` working
- mini_assistant/tiny_backend.py loading `09_chat.py` working
"""

from legacy_chat import *  # noqa: F401,F403


if __name__ == "__main__":
    main()

