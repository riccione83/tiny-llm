#!/usr/bin/env python3
"""
00_start.py

Simple terminal menu to run the pipeline steps.
"""

import os
import subprocess
import sys


MENU = """
Select an option:
  1) Build chat corpus + tokenize (01_make_chat_corpus_and_tokenize.py)
  2) Train base chat model (02_train_base_chat.py)
  3) Build instruct dataset (03_create_instruct.py)
  4) Small synthetic chat set (04_make_synthetic_chat.py)
  5) Large synthetic SFT set (05_make_synth_chat_sft.py)
  6) Build feedback SFT set (06_make_feedback_sft.py)
  7) LoRA fine-tune (07_lora_and_chat.py --mode lora)
  8) Synthetic LoRA (07_lora_and_chat.py --mode synth_lora)
  9) Feedback LoRA (07_lora_and_chat.py --mode feedback_lora)
  10) Chat (07_lora_and_chat.py --mode chat --use_lora)
  11) Create instruct v2 (10_create_instruct_v2.py)
  12) Train base v2 (11_train_base_v2.py)
  13) Exit
"""


def run(cmd):
    print(f"\n> {cmd}")
    return subprocess.call(cmd, shell=True)


def main():
    while True:
        print(MENU)
        choice = input("Enter choice: ").strip()

        if choice == "1":
            run(f"{sys.executable} .\\01_make_chat_corpus_and_tokenize.py")
        elif choice == "2":
            run(f"{sys.executable} .\\02_train_base_chat.py")
        elif choice == "3":
            run(f"{sys.executable} .\\03_create_instruct.py")
        elif choice == "4":
            run(f"{sys.executable} .\\04_make_synthetic_chat.py")
        elif choice == "5":
            run(f"{sys.executable} .\\05_make_synth_chat_sft.py")
        elif choice == "6":
            run(f"{sys.executable} .\\06_make_feedback_sft.py")
        elif choice == "7":
            run(f"{sys.executable} .\\07_lora_and_chat.py --mode lora")
        elif choice == "8":
            run(f"{sys.executable} .\\07_lora_and_chat.py --mode synth_lora")
        elif choice == "9":
            run(f"{sys.executable} .\\07_lora_and_chat.py --mode feedback_lora")
        elif choice == "10":
            run(f"{sys.executable} .\\07_lora_and_chat.py --mode chat --use_lora")
        elif choice == "11":
            run(f"{sys.executable} .\\10_create_instruct_v2.py")
        elif choice == "12":
            run(f"{sys.executable} .\\11_train_base_v2.py")
        elif choice == "13":
            print("Bye.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
