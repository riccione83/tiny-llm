#!/usr/bin/env python3
"""
Build targeted hard-case SFT data for micro-retraining.

Output JSONL rows:
  {"instruction": "User: ...\\nAssistant:", "output": "..."}

Focus:
- arithmetic with strict "number only" outputs
- YES/NO constraint prompts
- short/noisy text summarization (including typo-like inputs)
- web-style summarization (title/date/nav/footer noise, multi-snippet notes)
- strict templates:
    * "Summarize in exactly 2 sentences and do not add any information..."
    * "Summarize as exactly 2 bullet points: ..."
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Tuple


def norm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def maybe_typo(s: str, rng: random.Random, p: float = 0.15) -> str:
    if rng.random() > p:
        return s
    swaps = {
        "suddenly": "suddently",
        "distracted": "distrated",
        "because": "becouse",
        "their": "thier",
        "friend": "freind",
        "completed": "compleated",
        "before": "beofre",
    }
    out = s
    for a, b in swaps.items():
        if rng.random() < 0.35:
            out = re.sub(rf"\b{re.escape(a)}\b", b, out, flags=re.IGNORECASE)
    return out


def make_math_rows(rng: random.Random, n: int) -> List[Tuple[str, str]]:
    if n <= 0:
        return []
    rows: List[Tuple[str, str]] = []
    fixed = [
        ("What is 81/9?", "9"),
        ("What is 20/5?", "4"),
        ("What is 12*7?", "84"),
        ("What is 100-37?", "63"),
    ]
    for q, a in fixed[: min(len(fixed), n)]:
        rows.append((f"User: {q} Answer with a number only.\nAssistant:", a))

    ops = ["+", "-", "*", "/"]
    while len(rows) < n:
        op = rng.choice(ops)
        if op == "+":
            a = rng.randint(0, 200)
            b = rng.randint(0, 200)
            ans = a + b
        elif op == "-":
            a = rng.randint(0, 300)
            b = rng.randint(0, a)
            ans = a - b
        elif op == "*":
            a = rng.randint(0, 30)
            b = rng.randint(0, 30)
            ans = a * b
        else:
            b = rng.randint(1, 20)
            q = rng.randint(0, 40)
            a = b * q
            ans = q

        prompt = f"User: What is {a}{op}{b}? Answer with a number only.\nAssistant:"
        rows.append((prompt, str(ans)))
    return rows


def make_yesno_rows(rng: random.Random, n: int) -> List[Tuple[str, str]]:
    if n <= 0:
        return []
    rows: List[Tuple[str, str]] = []
    facts = [
        ("Is Rome in Italy?", "YES"),
        ("Is Tokyo in Japan?", "YES"),
        ("Is 2+2=5?", "NO"),
        ("Is water dry?", "NO"),
        ("Is Paris in France?", "YES"),
        ("Is 10 greater than 50?", "NO"),
    ]
    for q, a in facts[: min(len(facts), n)]:
        rows.append((f"User: Reply YES or NO only: {q}\nAssistant:", a))

    while len(rows) < n:
        a = rng.randint(0, 40)
        b = rng.randint(0, 40)
        if rng.random() < 0.5:
            c = a + b
            correct = "YES"
        else:
            c = a + b + rng.choice([-3, -2, -1, 1, 2, 3])
            correct = "NO"
        q = f"Is {a}+{b}={c}?"
        rows.append((f"User: Reply YES or NO only: {q}\nAssistant:", correct))
    return rows


def make_short_sum_rows(
    rng: random.Random,
    n: int,
    typo_prob: float = 0.15,
    bullet_prob: float = 0.0,
    strict_prob: float = 0.0,
) -> List[Tuple[str, str]]:
    if n <= 0:
        return []
    rows: List[Tuple[str, str]] = []
    names = ["Sara", "Luca", "Mina", "Jon", "Elena", "Marco", "Ava", "Noah"]
    tasks = ["history", "math", "biology", "physics", "grammar", "coding"]
    interruptions = ["friend knocked at the door", "phone rang", "dog started barking", "power went out"]
    outcomes = [
        "she finished late but completed all exercises",
        "he paused and then completed most of the lesson",
        "she resumed after ten minutes and finished the chapter",
        "he lost focus and completed only half the tasks",
    ]

    strict_tpl = "User: Summarize in exactly 2 sentences and do not add any information not present in the text: {text}\nAssistant:"
    plain_tpl = "User: Summarize in 2 sentences: {text}\nAssistant:"
    bullet_tpl = "User: Summarize as exactly 2 bullet points: {text}\nAssistant:"

    while len(rows) < n:
        name = rng.choice(names)
        subject = rng.choice(tasks)
        intr = rng.choice(interruptions)
        out = rng.choice(outcomes)
        text = f"{name} was studying {subject} alone when {intr}. After the interruption, {out}."
        text = maybe_typo(text, rng, p=typo_prob)

        s1 = f"{name} was studying {subject} alone but was interrupted when {intr}."
        s2 = out[:1].upper() + out[1:] + "."
        summary_2sent = f"{s1} {s2}"

        mode = rng.random()
        if mode < bullet_prob:
            prompt = bullet_tpl.format(text=text)
            b1 = f"- {name} was studying {subject} alone and was interrupted when {intr}."
            b2 = f"- {out[:1].upper() + out[1:]}."
            output = f"{b1}\n{b2}"
        elif mode < (bullet_prob + strict_prob):
            prompt = strict_tpl.format(text=text)
            output = summary_2sent
        else:
            prompt = plain_tpl.format(text=text)
            output = summary_2sent

        rows.append((prompt, output))
    return rows


def make_web_sum_rows(
    rng: random.Random,
    n: int,
    bullet_prob: float = 0.0,
    strict_prob: float = 0.0,
) -> List[Tuple[str, str]]:
    if n <= 0:
        return []
    rows: List[Tuple[str, str]] = []

    pages = [
        {
            "title": "NVIDIA announces new RTX 60-series GPUs",
            "date": "2026-01-18",
            "source": "techdaily.example",
            "body": (
                "NVIDIA introduced RTX 6090 and RTX 6080 desktop GPUs at its January event. "
                "The company said the RTX 6090 will launch first, followed by the RTX 6080 two weeks later. "
                "NVIDIA also announced improved ray tracing performance and lower power draw than the previous generation."
            ),
            "summary2": (
                "NVIDIA announced RTX 6090 and RTX 6080 desktop GPUs at a January event, with the 6090 launching first and the 6080 two weeks later. "
                "The company says the new cards improve ray tracing performance while using less power than the previous generation."
            ),
            "bullets": (
                "- NVIDIA announced RTX 6090 and RTX 6080 desktop GPUs, with staggered launch timing.\n"
                "- NVIDIA says the new generation improves ray tracing and power efficiency."
            ),
        },
        {
            "title": "City opens new electric bus line",
            "date": "2025-11-03",
            "source": "metro-news.example",
            "body": (
                "The city launched a new electric bus line connecting the airport and downtown. "
                "Officials said buses run every 12 minutes during peak hours. "
                "The transport agency expects the line to reduce diesel bus traffic on the route."
            ),
            "summary2": (
                "The city opened a new electric bus line between the airport and downtown. "
                "Buses run every 12 minutes at peak times, and officials expect less diesel traffic on that route."
            ),
            "bullets": (
                "- A new electric bus line now connects the airport and downtown.\n"
                "- Service runs every 12 minutes during peak hours and is expected to reduce diesel traffic."
            ),
        },
        {
            "title": "Hospital deploys AI triage pilot",
            "date": "2025-09-22",
            "source": "healthwire.example",
            "body": (
                "A regional hospital started a six-month AI triage pilot in its emergency department. "
                "The pilot suggests priority levels but final decisions remain with clinicians. "
                "Hospital leaders said they will evaluate waiting times and safety metrics before expanding."
            ),
            "summary2": (
                "A regional hospital began a six-month AI triage pilot in emergency care. "
                "The system suggests priorities while clinicians keep final authority, and the hospital will review wait-time and safety results before expansion."
            ),
            "bullets": (
                "- A hospital launched a six-month AI triage pilot in its emergency department.\n"
                "- Clinicians keep final decisions, and the hospital will evaluate wait-time and safety outcomes before any expansion."
            ),
        },
        {
            "title": "University lowers tuition for online program",
            "date": "2025-07-10",
            "source": "campus-report.example",
            "body": (
                "A public university reduced tuition for its online data science program by 15 percent. "
                "The change applies to new students starting in the fall term. "
                "Administrators said the lower price is intended to improve access for working professionals."
            ),
            "summary2": (
                "A public university cut tuition by 15 percent for its online data science program for new fall entrants. "
                "Administrators said the reduction is meant to improve access for working professionals."
            ),
            "bullets": (
                "- The university reduced online data science tuition by 15 percent for new fall students.\n"
                "- Administrators said the goal is better access for working professionals."
            ),
        },
    ]

    plain_templates = [
        "User: Summarize in 2 sentences: {text}\nAssistant:",
        "User: Cosa dice questo articolo? {text}\nAssistant:",
        "User: Summarize this web page in 2 sentences: {text}\nAssistant:",
    ]
    strict_template = "User: Summarize in exactly 2 sentences and do not add any information not present in the text: {text}\nAssistant:"
    bullet_template = "User: Summarize as exactly 2 bullet points: {text}\nAssistant:"

    while len(rows) < n:
        p = rng.choice(pages)
        noisy = (
            "Home | News | Reviews | Contact\n"
            f"Published: {p['date']}\n"
            f"Source: {p['source']}\n\n"
            f"Title: {p['title']}\n\n"
            f"{p['body']}\n\n"
            "Related articles | Privacy Policy | Terms"
        )

        if rng.random() < 0.25:
            # Simulate a "latest" style query built from short snippets.
            noisy = (
                "Search query: latest nvidia gpus\n"
                "Snippet 1: NVIDIA announced RTX 6090 and RTX 6080 desktop GPUs at a January event.\n"
                "Snippet 2: The RTX 6090 launches first; RTX 6080 follows two weeks later.\n"
                "Snippet 3: NVIDIA says the new generation improves ray tracing and lowers power draw."
            )

        if rng.random() < bullet_prob:
            prompt = bullet_template.format(text=noisy)
            output = p["bullets"]
        else:
            if rng.random() < strict_prob:
                prompt = strict_template.format(text=noisy)
            else:
                prompt = rng.choice(plain_templates).format(text=noisy)
            output = p["summary2"]

        rows.append((prompt, output))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hard_cases_sft.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_math", type=int, default=6000)
    ap.add_argument("--n_yesno", type=int, default=2000)
    ap.add_argument("--n_shortsum", type=int, default=6000)
    ap.add_argument("--n_websum", type=int, default=4000)
    ap.add_argument("--typo_prob", type=float, default=0.15)
    ap.add_argument("--shortsum_bullet_prob", type=float, default=0.0)
    ap.add_argument("--websum_bullet_prob", type=float, default=0.0)
    ap.add_argument("--shortsum_strict_prob", type=float, default=0.0)
    ap.add_argument("--websum_strict_prob", type=float, default=0.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, str]] = []
    rows.extend(make_math_rows(rng, args.n_math))
    rows.extend(make_yesno_rows(rng, args.n_yesno))
    rows.extend(
        make_short_sum_rows(
            rng,
            args.n_shortsum,
            typo_prob=float(args.typo_prob),
            bullet_prob=float(args.shortsum_bullet_prob),
            strict_prob=float(args.shortsum_strict_prob),
        )
    )
    rows.extend(
        make_web_sum_rows(
            rng,
            args.n_websum,
            bullet_prob=float(args.websum_bullet_prob),
            strict_prob=float(args.websum_strict_prob),
        )
    )
    rng.shuffle(rows)

    with out_path.open("w", encoding="utf-8") as f:
        for ins, out in rows:
            rec = {"instruction": norm(ins), "output": norm(out)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Saved: {out_path} | rows={len(rows):,} "
        f"(math={args.n_math:,}, yesno={args.n_yesno:,}, shortsum={args.n_shortsum:,}, websum={args.n_websum:,})"
    )


if __name__ == "__main__":
    main()
