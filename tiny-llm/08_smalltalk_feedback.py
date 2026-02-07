#!/usr/bin/env python3
"""
Create a small targeted feedback set for greetings + basic definitions.
Appends to feedback/feedback_sft.jsonl
"""
import json
import os
from pathlib import Path

OUT = Path("feedback/feedback_sft.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

pairs = []

# greetings / smalltalk
pairs += [
    ("Hi", "Hi there! How can I help you today?"),
    ("Hello", "Hello! What would you like to talk about?"),
    ("Hey", "Hey! How can I help?"),
    ("How are you?", "I’m doing well, thanks for asking! How are you?"),
    ("What's up?", "Not much, just here to help. What's up with you?"),
    ("Good morning", "Good morning! Anything I can help with?"),
    ("Good evening", "Good evening! How can I help?"),
    ("Thanks", "You’re welcome!"),
    ("Thank you", "You’re welcome!"),
    ("Bye", "Bye! Have a great day."),
]

# basic definitions
pairs += [
    ("What is software?", "Software is a set of programs and data that tell a computer what to do."),
    ("What is hardware?", "Hardware is the physical parts of a computer, like the CPU, memory, and storage."),
    ("What is the internet?", "The internet is a global network that connects computers and devices."),
    ("What is a website?", "A website is a collection of web pages available on the internet."),
    ("What is an operating system?", "An operating system manages hardware and runs applications."),
    ("What is a browser?", "A browser is a program used to access and view websites."),
    ("What is programming?", "Programming is writing instructions that tell computers how to perform tasks."),
    ("What is an algorithm?", "An algorithm is a step-by-step method for solving a problem."),
    ("What is data?", "Data is information that can be processed or stored."),
    ("What is AI?", "AI is software that performs tasks that normally require human intelligence."),
]

# expand with slight variants
expanded = []
for q, a in pairs:
    expanded.append((q, a))
    if q.endswith("?"):
        expanded.append((q.replace("?", ""), a))
    if q.lower().startswith("what is "):
        expanded.append((q.replace("What is", "Define", 1), a))

rows = [{"instruction": q, "chosen": a} for q, a in expanded]

with OUT.open("a", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"added={len(rows)}")
