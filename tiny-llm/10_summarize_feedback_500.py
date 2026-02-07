#!/usr/bin/env python3
"""
Append ~500 summarization examples to feedback/feedback_sft.jsonl.
"""
import json
import random
from pathlib import Path

OUT = Path("feedback/feedback_sft.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)

subjects = [
    "solar energy", "photosynthesis", "gravity", "electricity", "the internet",
    "machine learning", "climate change", "vaccines", "batteries", "semiconductors",
    "rainfall", "earthquakes", "clouds", "ocean currents", "wildfires",
    "software development", "databases", "operating systems", "cybersecurity", "networks",
]

templates = [
    ("Summarize this text: {text}", "{summary}"),
    ("Summarize: {text}", "{summary}"),
    ("Please summarize: {text}", "{summary}"),
]

facts = {
    "solar energy": (
        "Solar energy uses sunlight captured by panels to generate electricity. It is renewable and produces no direct emissions.",
        "Solar energy uses sunlight to generate electricity and is a clean, renewable source."
    ),
    "photosynthesis": (
        "Photosynthesis lets plants convert sunlight, water, and carbon dioxide into sugars and oxygen. It happens mainly in leaves.",
        "Photosynthesis is how plants use sunlight, water, and CO2 to make sugars and release oxygen."
    ),
    "gravity": (
        "Gravity is the force that attracts objects with mass toward each other. On Earth it gives weight and causes objects to fall.",
        "Gravity attracts masses; on Earth it gives weight and makes objects fall."
    ),
    "electricity": (
        "Electricity is the flow of electric charge through a conductor. It powers devices and can be generated in many ways.",
        "Electricity is the flow of electric charge that powers devices and can be produced from various sources."
    ),
    "the internet": (
        "The internet is a global network of networks that lets devices communicate using standard protocols. It enables services like the web and email.",
        "The internet is a global network enabling device communication and services like the web and email."
    ),
    "machine learning": (
        "Machine learning is a field of AI where models learn patterns from data to make predictions or decisions.",
        "Machine learning trains models on data to make predictions or decisions."
    ),
    "climate change": (
        "Climate change refers to long-term shifts in temperature and weather patterns, largely driven by human emissions of greenhouse gases.",
        "Climate change is long-term warming and weather shifts, largely driven by greenhouse gas emissions."
    ),
    "vaccines": (
        "Vaccines train the immune system to recognize and fight pathogens, helping prevent disease.",
        "Vaccines prepare the immune system to prevent or reduce disease from pathogens."
    ),
    "batteries": (
        "Batteries store energy chemically and release it as electricity when needed. They power portable devices and vehicles.",
        "Batteries store chemical energy and deliver it as electricity for devices and vehicles."
    ),
    "semiconductors": (
        "Semiconductors are materials with controllable conductivity, forming the basis of modern electronics like chips and transistors.",
        "Semiconductors are materials with controllable conductivity used in chips and electronics."
    ),
    "rainfall": (
        "Rainfall occurs when water vapor condenses into droplets that become heavy enough to fall from clouds.",
        "Rainfall happens when condensed water droplets in clouds become heavy and fall."
    ),
    "earthquakes": (
        "Earthquakes are sudden ground movements caused by shifts along faults in Earth's crust.",
        "Earthquakes are sudden ground movements from fault shifts in the Earth's crust."
    ),
    "clouds": (
        "Clouds form when moist air cools and water vapor condenses into tiny droplets or ice crystals.",
        "Clouds form as moist air cools and vapor condenses into droplets or ice."
    ),
    "ocean currents": (
        "Ocean currents are large-scale flows driven by wind, temperature, and salinity differences, affecting climate and ecosystems.",
        "Ocean currents are large-scale flows driven by wind and density that affect climate and ecosystems."
    ),
    "wildfires": (
        "Wildfires are uncontrolled fires in vegetation fueled by dry conditions, wind, and heat.",
        "Wildfires are uncontrolled vegetation fires fueled by dry conditions, wind, and heat."
    ),
    "software development": (
        "Software development is the process of designing, building, testing, and maintaining software applications.",
        "Software development involves designing, building, testing, and maintaining applications."
    ),
    "databases": (
        "Databases store and organize data so it can be queried, updated, and managed efficiently.",
        "Databases organize data so it can be stored, queried, and managed efficiently."
    ),
    "operating systems": (
        "An operating system manages hardware resources and provides services for applications.",
        "An operating system manages hardware and provides services for applications."
    ),
    "cybersecurity": (
        "Cybersecurity protects systems and data from attacks, unauthorized access, and damage.",
        "Cybersecurity protects systems and data from attacks and unauthorized access."
    ),
    "networks": (
        "Computer networks connect devices so they can share data and resources over wired or wireless links.",
        "Networks connect devices to share data and resources over wired or wireless links."
    ),
}

rows = []
for _ in range(500):
    topic = random.choice(subjects)
    text, summary = facts[topic]
    tpl, out_tpl = random.choice(templates)
    rows.append({"instruction": tpl.format(text=text), "chosen": out_tpl.format(summary=summary)})

with OUT.open("a", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"added={len(rows)}")
