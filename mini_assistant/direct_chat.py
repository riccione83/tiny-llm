#!/usr/bin/env python3
"""
Plain local chat (no web grounding).

This is the tool you want for quick qualitative testing of a locally trained
HF model directory (or a HF repo id). For the grounded/web assistant, use:
`python -m mini_assistant.chat`.
"""

from __future__ import annotations

import argparse

from .llm import LocalLLM


def main() -> None:
    ap = argparse.ArgumentParser(description="Mini direct chat (no web)")
    ap.add_argument("--backend", default="hf", choices=["hf", "tiny"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--tiny_ckpt", default="checkpoints_v2/final.pt")
    ap.add_argument("--tiny_tokenizer", default="tokenizer.model")
    ap.add_argument("--tiny_lora", default="")
    ap.add_argument("--tiny_top_p", type=float, default=1.0)
    ap.add_argument("--system", default="", help="Optional system prompt")
    ap.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable tokenizer.apply_chat_template (use plain role-prefixed text). Useful for non-instruct base models.",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=240)
    args = ap.parse_args()

    llm = LocalLLM(
        backend=args.backend,
        model_name=args.model_name,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        tiny_ckpt=args.tiny_ckpt,
        tiny_tokenizer=args.tiny_tokenizer,
        tiny_lora=args.tiny_lora,
        tiny_top_p=float(args.tiny_top_p),
        use_chat_template=(not bool(args.no_chat_template)),
    )

    messages = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    print("Mini Direct Chat ready. Commands: /exit, /reset")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        low = user.lower()
        if low in {"/exit", "exit", "quit", "/quit"}:
            print("Bye.")
            break
        if low in {"/reset", "/clear"}:
            messages = []
            if args.system.strip():
                messages.append({"role": "system", "content": args.system.strip()})
            print("Chat reset.")
            continue

        messages.append({"role": "user", "content": user})
        ans = llm.generate(messages)
        messages.append({"role": "assistant", "content": ans})
        print(f"Bot: {ans}\n")


if __name__ == "__main__":
    main()
