#!/usr/bin/env python3
import argparse

from .config import AssistantConfig
from .engine import GroundedWebAssistant
from .web import is_probable_url


def main() -> None:
    ap = argparse.ArgumentParser(description="Mini grounded web assistant")
    ap.add_argument("--url", default="", help="Optional fixed source URL for all questions")
    ap.add_argument("--backend", default="hf", choices=["hf", "tiny"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tiny_ckpt", default="checkpoints_v2/final.pt")
    ap.add_argument("--tiny_tokenizer", default="tokenizer.model")
    ap.add_argument("--tiny_lora", default="")
    ap.add_argument("--tiny_top_p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--search_results", type=int, default=5)
    ap.add_argument("--timeout_sec", type=int, default=20)
    ap.add_argument("--direct_confidence_threshold", type=float, default=0.72)
    ap.add_argument("--show_debug", action="store_true")
    args = ap.parse_args()

    cfg = AssistantConfig(
        backend=args.backend,
        llm_model_name=args.model_name,
        embedding_model_name=args.embedding_model,
        tiny_ckpt=args.tiny_ckpt,
        tiny_tokenizer=args.tiny_tokenizer,
        tiny_lora=args.tiny_lora,
        tiny_top_p=float(args.tiny_top_p),
        temperature=float(args.temperature),
        max_new_tokens=int(args.max_new_tokens),
        top_k=int(args.top_k),
        search_results=int(args.search_results),
        timeout_sec=int(args.timeout_sec),
        direct_confidence_threshold=float(args.direct_confidence_threshold),
    )
    assistant = GroundedWebAssistant(cfg)
    current_url = args.url.strip()

    print("Mini Assistant ready. Commands: /exit, /url <link>, /clearurl, /showurl")
    if current_url:
        print(f"Fixed URL: {current_url}")

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
        if low.startswith("/url "):
            current_url = user.split(" ", 1)[1].strip()
            print(f"URL set: {current_url}")
            continue
        if low == "/clearurl":
            current_url = ""
            print("URL cleared.")
            continue
        if low == "/showurl":
            print(f"Current URL: {current_url or '(none)'}")
            continue

        # Friendly shortcut: if user types a URL directly.
        if is_probable_url(user) and not current_url:
            current_url = user
            print(f"URL set: {current_url}")
            print("Now ask your question.")
            continue

        res = assistant.answer(user, url=current_url, search_if_missing=(not bool(current_url)))
        print(f"Bot: {res.answer}")
        if args.show_debug:
            print(f"Route: {res.debug.get('route', '?')} | Debug: {res.debug}")
        # if res.sources:
        #     print("Sources:")
        #     for s in res.sources:
        #         print(f"- {s}")
        print("")


if __name__ == "__main__":
    main()
