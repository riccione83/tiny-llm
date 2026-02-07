#!/usr/bin/env python3
"""
04_make_synthetic_chat.py

Generate a small synthetic chat SFT dataset (JSONL) with fields:
  - instruction
  - output

Outputs:
  data/synth_chat_sft.jsonl
"""

from pathlib import Path
import json
import random

OUT_PATH = Path("data/synth_chat_sft.jsonl")
OUT_PATH.parent.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)

COUNTS = {
    "greetings": 300,
    "motivation": 400,
    "stress": 400,
    "small_talk": 400,
    "arithmetic": 800,
    "capitals": 600,
    "definitions": 400,
}

GREETINGS = [
    ("Hi!", "Hi there! How can I help today?"),
    ("Hello!", "Hello! What would you like to chat about?"),
    ("Hey", "Hey! How's your day going?"),
    ("Good morning", "Good morning! Anything you'd like to work on?"),
    ("Good evening", "Good evening! How can I help?"),
]

MOTIVATION = [
    ("I feel unmotivated.", "Try a 5‑minute start: pick one tiny step and do it now."),
    ("I can't focus today.", "Set a 10‑minute timer and aim for just one task."),
    ("How do I stay consistent?", "Keep the goal small and tie it to a daily routine."),
    ("I'm procrastinating.", "Make the task easier, then do the first 2 minutes."),
    ("I need motivation to study.", "Break it into short sessions and reward each one."),
]

STRESS = [
    ("I'm stressed.", "Take five slow breaths and unclench your jaw and shoulders."),
    ("How can I relax quickly?", "Try box breathing: 4 in, hold 4, out 4, hold 4."),
    ("I'm anxious before sleep.", "Write a quick worry list and park it for tomorrow."),
    ("Any quick stress tips?", "Stretch for 60 seconds and drink a glass of water."),
    ("I'm overwhelmed.", "Pick the single most important next step and do only that."),
]

SMALL_TALK = [
    ("How are you?", "I'm here and ready to help. How are you doing?"),
    ("What's up?", "Not much, just here to chat. What's up with you?"),
    ("How's it going?", "Pretty good! Want to talk about anything fun?"),
    ("Do you like music?", "I can talk about music—what do you like?"),
    ("Any plans for the weekend?", "No plans here—what are you hoping to do?"),
]

ARITHMETIC = [
    ("What is {a} + {b}?", "{a} + {b} = {c}."),
    ("Calculate {a} - {b}.", "{a} - {b} = {c}."),
    ("What's {a} * {b}?", "{a} * {b} = {c}."),
    ("Compute {a} / {b}.", "{a} / {b} = {c}."),
]

CAPITALS = {
    "France": "Paris",
    "Germany": "Berlin",
    "Italy": "Rome",
    "Spain": "Madrid",
    "United Kingdom": "London",
    "Canada": "Ottawa",
    "Japan": "Tokyo",
    "South Korea": "Seoul",
    "Australia": "Canberra",
    "Brazil": "Brasilia",
    "Mexico": "Mexico City",
    "Argentina": "Buenos Aires",
    "India": "New Delhi",
    "China": "Beijing",
    "Egypt": "Cairo",
    "Nigeria": "Abuja",
    "Kenya": "Nairobi",
    "South Africa": "Pretoria",
    "Turkey": "Ankara",
    "Saudi Arabia": "Riyadh",
}

DEFINITIONS = [
    ("What is a transformer model?", "A transformer is a neural network that uses attention to process sequences."),
    ("Define gradient descent.", "Gradient descent is an optimization method that steps downhill on a loss curve."),
    ("What is attention in ML?", "Attention lets a model focus on the most relevant parts of the input."),
    ("What is overfitting?", "Overfitting is when a model memorizes training data and generalizes poorly."),
    ("What is a loss function?", "A loss function measures how wrong a model's predictions are."),
]

def add_many(rows, target_count, template_pairs):
    while len(rows) < target_count:
        instr, out = random.choice(template_pairs)
        rows.append({"instruction": instr, "output": out})

def add_arithmetic(rows, target_count):
    while len(rows) < target_count:
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        op = random.choice(ARITHMETIC)
        if " / " in op[0]:
            b = random.randint(1, 12)
            a = b * random.randint(1, 12)
            c = a // b
        elif " - " in op[0]:
            a, b = max(a, b), min(a, b)
            c = a - b
        elif " * " in op[0]:
            c = a * b
        else:
            c = a + b
        instr = op[0].format(a=a, b=b)
        out = op[1].format(a=a, b=b, c=c)
        rows.append({"instruction": instr, "output": out})

def add_capitals(rows, target_count):
    countries = list(CAPITALS.items())
    while len(rows) < target_count:
        country, capital = random.choice(countries)
        instr = f"What is the capital of {country}?"
        out = f"The capital of {country} is {capital}."
        rows.append({"instruction": instr, "output": out})

def main():
    rows = []

    add_many(rows, COUNTS["greetings"], GREETINGS)
    add_many(rows, len(rows) + COUNTS["motivation"], MOTIVATION)
    add_many(rows, len(rows) + COUNTS["stress"], STRESS)
    add_many(rows, len(rows) + COUNTS["small_talk"], SMALL_TALK)
    add_arithmetic(rows, len(rows) + COUNTS["arithmetic"])
    add_capitals(rows, len(rows) + COUNTS["capitals"])
    add_many(rows, len(rows) + COUNTS["definitions"], DEFINITIONS)

    random.shuffle(rows)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {OUT_PATH}")

if __name__ == "__main__":
    main()
