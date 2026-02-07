#!/usr/bin/env python3
"""
Append a small summarization feedback set to feedback/feedback_sft.jsonl.
"""
import json
from pathlib import Path

OUT = Path("feedback/feedback_sft.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

examples = [
    (
        "Summarize this text: The sun is a star at the center of our solar system. It provides light and heat that make life on Earth possible. It is mostly made of hydrogen and helium.",
        "The sun is a hydrogen- and helium-rich star at the center of our solar system that provides light and heat for life on Earth."
    ),
    (
        "Summarize this text: Photosynthesis lets plants turn sunlight, water, and carbon dioxide into sugars and oxygen. It happens mainly in leaves.",
        "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make sugars and release oxygen, mostly in leaves."
    ),
    (
        "Summarize this text: A transformer transfers electrical energy between circuits using electromagnetic induction. Power transformers increase or decrease voltage.",
        "A transformer transfers electrical energy between circuits via electromagnetic induction and can raise or lower voltage."
    ),
    (
        "Summarize this text: The internet is a global network of networks that allows devices to communicate using standardized protocols. It enables services like the web and email.",
        "The internet is a global network of networks that lets devices communicate using standard protocols, enabling services like the web and email."
    ),
    (
        "Summarize this text: Gravity is the force that attracts objects with mass toward each other. On Earth, it gives weight and causes objects to fall.",
        "Gravity is the force that attracts masses; on Earth it gives weight and makes objects fall."
    ),
]

# Expand with small variations
rows = []
for q, a in examples:
    rows.append({"instruction": q, "chosen": a})
    rows.append({"instruction": q.replace("Summarize this text:", "Summarize:"), "chosen": a})
    rows.append({"instruction": q.replace("Summarize this text:", "Please summarize:"), "chosen": a})

with OUT.open("a", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"added={len(rows)}")
