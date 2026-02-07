#!/usr/bin/env python3
"""
Auto-generate lots of feedback rows (math/science/chat/tech),
optionally run the model once per prompt (for BEFORE preview),
and append canonical answers to feedback/feedback_sft.jsonl.

Usage:
  python 08_autofeedback.py
"""

import json
import os
import random
import re
import importlib.util
from pathlib import Path

import torch
import sentencepiece as spm

SEED = 42

# Increase this as you want: 1_000, 5_000, 10_000, 50_000...
TARGET = 10_000

# Only used for preview prints; dataset generation does not need model calls.
RUN_MODEL_FOR_REPORT = True
REPORT_N = 12  # how many BEFORE/AFTER to print

random.seed(SEED)

mod_path = Path(__file__).with_name("07_lora_and_chat.py")
spec = importlib.util.spec_from_file_location("lc", mod_path)
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


# -------------------------
# Helpers
# -------------------------
_math_re = re.compile(r"What is (\-?\d+)\s*([+\-*/])\s*(\-?\d+)\?")

def math_expected(prompt: str):
    m = _math_re.match(prompt)
    if not m:
        return None
    a = int(m.group(1))
    op = m.group(2)
    b = int(m.group(3))
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    if op == "/":
        if b == 0:
            return None
        # prefer integer when exact
        if a % b == 0:
            return str(a // b)
        # short float (avoid long repeating)
        return f"{a / b:.4g}"
    return None


def build_math_prompts():
    prompts = []

    # Additions/subtractions (incl negatives)
    for a in range(-50, 201):
        b = random.randint(-50, 50)
        prompts.append(f"What is {a}+{b}?")
        prompts.append(f"What is {a}-{b}?")

    # Multiplications (small-ish for clean answers)
    for a in range(-30, 101):
        for b in [2, 3, 4, 5, 6, 7, 8, 9]:
            prompts.append(f"What is {a}*{b}?")

    # Divisions that are often exact
    divisors = [2, 3, 4, 5, 6, 8, 10, 12]
    for b in divisors:
        for k in range(-50, 201):
            a = k * b
            prompts.append(f"What is {a}/{b}?")

    # A few non-exact divisions (still short)
    for _ in range(400):
        a = random.randint(-200, 200)
        b = random.choice([3, 7, 9, 11])
        if b != 0:
            prompts.append(f"What is {a}/{b}?")

    return prompts


def build_science_bank():
    # Keep answers short, canonical, chat-friendly (1–2 sentences)
    science = {
        "What is photosynthesis?": "Photosynthesis is how plants use sunlight to make sugars from water and carbon dioxide, releasing oxygen.",
        "Why is the sky blue?": "The sky looks blue because shorter blue wavelengths of sunlight scatter more in the atmosphere.",
        "What is gravity?": "Gravity is the force that attracts masses toward each other, like Earth pulling objects down.",
        "What are the states of matter?": "The main states are solid, liquid, gas, and plasma.",
        "What is an atom?": "An atom is the smallest unit of an element, made of protons, neutrons, and electrons.",
        "What is DNA?": "DNA is the molecule that carries genetic instructions for living organisms.",
        "Why do we have seasons?": "Seasons are caused by Earth's axial tilt as it orbits the Sun.",
        "What is evaporation?": "Evaporation is when liquid turns into vapor at the surface of a liquid.",
        "What is the water cycle?": "The water cycle moves water through evaporation, condensation, and precipitation.",
        "What is energy?": "Energy is the ability to do work or cause change.",
        "What is a cell?": "A cell is the basic unit of life, and organisms are made of one or more cells.",
        "What is an ecosystem?": "An ecosystem is a community of living things interacting with each other and their environment.",
        "What is a planet?": "A planet is a large body that orbits a star and is massive enough to be nearly round.",
        "What is a black hole?": "A black hole is a region where gravity is so strong that not even light can escape.",
        "What is an electric current?": "Electric current is the flow of electric charge, usually carried by electrons in a conductor.",
        "What is voltage?": "Voltage is the electrical potential difference that pushes current through a circuit.",
        "What is a chemical reaction?": "A chemical reaction rearranges atoms to form new substances.",
        "What is pH?": "pH measures how acidic or basic a solution is, on a scale from 0 to 14.",
        "What is a virus?": "A virus is a tiny infectious agent that replicates only inside living cells.",
        "What is a vaccine?": "A vaccine trains the immune system to recognize and fight a specific disease.",
    }

    # Expand with templates (still canonical)
    template_defs = {
        "What is {term}?": {
            "inertia": "Inertia is the tendency of an object to resist changes in its motion.",
            "friction": "Friction is a force that resists motion between two surfaces in contact.",
            "density": "Density is mass per unit volume—how much matter is packed into a space.",
            "acceleration": "Acceleration is how quickly velocity changes over time.",
            "velocity": "Velocity is speed with a direction.",
            "mass": "Mass is the amount of matter in an object.",
            "temperature": "Temperature measures how hot or cold something is, related to particle motion.",
            "heat": "Heat is energy transferred due to a temperature difference.",
            "conduction": "Conduction is heat transfer through direct contact.",
            "convection": "Convection is heat transfer by the movement of a fluid like air or water.",
            "radiation": "Radiation is energy transfer by electromagnetic waves, like sunlight.",
        }
    }

    for tpl, bank in template_defs.items():
        for term, ans in bank.items():
            q = tpl.format(term=term)
            science[q] = ans

    return science


def build_tech_bank():
    tech = {
        "What is a transformer model?": "A transformer is a neural network that uses attention to process sequences efficiently and in parallel.",
        "Explain gradient descent simply.": "Gradient descent minimizes loss by taking small steps in the direction that reduces error.",
        "What is overfitting?": "Overfitting is when a model learns training data too well and performs poorly on new data.",
        "What is attention in ML?": "Attention lets a model focus on the most relevant parts of the input when making a prediction.",
        "What is tokenization?": "Tokenization splits text into smaller units (tokens) for a model to process.",
        "What is a learning rate?": "The learning rate controls how big each update step is during training.",
        "What is dropout?": "Dropout randomly zeroes some activations during training to reduce overfitting.",
        "Explain big-O notation simply.": "Big-O describes how an algorithm's time or memory grows as input size increases.",
        "What is a neural network?": "A neural network is a model made of layers of weighted connections that learn patterns from data.",
        "What is backpropagation?": "Backpropagation computes gradients so weights can be updated to reduce loss.",
        "What is an embedding in ML?": "An embedding maps items (like words) to dense vectors that capture meaning.",
        "What is a loss function?": "A loss function measures how wrong a model's predictions are.",
        "What is a dataset split?": "A dataset split separates data into train/validation/test sets.",
        "What is a validation set?": "A validation set is held-out data used to tune and monitor model performance.",
        "What is regularization?": "Regularization reduces overfitting by discouraging overly complex models.",
        "What is a parameter in ML?": "A parameter is a learned value (like a weight) in the model.",
        "What is a gradient?": "A gradient tells you how loss changes when a parameter changes.",
        "What is an optimizer?": "An optimizer is the algorithm that updates model parameters to reduce loss.",
        "What is a batch?": "A batch is a small group of training examples processed together.",
        "What is an epoch?": "An epoch is one full pass through the training data.",
    }

    # Template expansions, mapped to canonical short answers
    term_bank = {
        "fine-tuning": "Fine-tuning continues training a pretrained model on a specific dataset to specialize its behavior.",
        "pretraining": "Pretraining learns general language patterns from lots of text before task-specific tuning.",
        "LoRA": "LoRA fine-tunes efficiently by learning low-rank updates to certain weight matrices.",
        "RMSNorm": "RMSNorm is a normalization method similar to LayerNorm but using RMS statistics.",
        "RoPE": "RoPE is a positional encoding method that rotates embeddings to represent token positions.",
        "KV cache": "A KV cache stores attention keys/values from previous tokens to speed up generation.",
    }

    templates = [
        "What is {term}?",
        "Explain {term} simply.",
        "Give a short definition of {term}.",
    ]

    for term, ans in term_bank.items():
        for tpl in templates:
            tech[tpl.format(term=term)] = ans

    return tech


def build_chat_bank():
    chat = {
        "Hi!": "Hi there! How can I help you today?",
        "Hello!": "Hello! What would you like to talk about?",
        "How are you?": "I'm doing well—thanks for asking! How are you?",
        "What's up?": "Not much—just here to help. What's up with you?",
        "Good morning": "Good morning! Anything I can help with?",
        "Good evening": "Good evening! How can I help?",
        "Thanks": "You're welcome!",
        "Thank you": "You're welcome!",
        "Bye": "Bye! Have a great day.",
        "I feel stressed. Any quick tips?": "Try box breathing: 4 in, hold 4, out 4, hold 4.",
        "I'm anxious. Any quick tips?": "Try grounding: name 5 things you see, 4 you feel, 3 you hear.",
        "I feel tired. Any advice?": "Try water, a short walk, and a 10-minute reset. Want a quick plan?",
        "Can you motivate me today?": "Pick one tiny task and do it for 5 minutes. Momentum will follow.",
        "Tell me a short joke.": "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "How can I focus better?": "Try 25 minutes focused work, then a 5-minute break. Want help picking the first task?",
        "What should I do after work to unwind?": "A short walk, a shower, or a light hobby helps—what sounds best right now?",
        "I'm overwhelmed. What should I do first?": "Pick the single most important next step and do just that—small is fine.",
    }

    # extra chat templates (still with canonical answers)
    chat_templates = {
        "I'm feeling {mood}. Any advice?": {
            "sad": "I'm sorry you're feeling that way. Try one small, kind action for yourself—want to talk about what's going on?",
            "stressed": "Try a slow breath: 4 in, 4 hold, 6 out. Want a 2-minute routine?",
            "angry": "Take a pause and breathe. A short walk or cold water can help reset—what triggered it?",
            "lonely": "That can feel heavy. Reaching out to one person or doing a small social activity can help—want ideas?",
        }
    }
    for tpl, moods in chat_templates.items():
        for mood, ans in moods.items():
            chat[tpl.format(mood=mood)] = ans

    return chat


def build_prompts():
    prompts = []

    # math prompts (lots)
    prompts.extend(build_math_prompts())

    # banks
    science = build_science_bank()
    tech = build_tech_bank()
    chat = build_chat_bank()

    prompts.extend(list(science.keys()))
    prompts.extend(list(tech.keys()))
    prompts.extend(list(chat.keys()))

    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    # deterministic shuffle for variety
    rng = random.Random(SEED)
    rng.shuffle(uniq)

    # cut to TARGET
    if len(uniq) >= TARGET:
        return uniq[:TARGET], science, tech, chat

    # if not enough, cycle (rare if TARGET <= a few 10k)
    out = []
    i = 0
    while len(out) < TARGET:
        out.append(uniq[i % len(uniq)])
        i += 1
    return out, science, tech, chat


def chosen_for_prompt(prompt: str, science: dict, tech: dict, chat: dict) -> str:
    expected = math_expected(prompt)
    if expected is not None:
        # Keep it short/chatty
        return f"{expected}."
    if prompt in science:
        return science[prompt]
    if prompt in tech:
        return tech[prompt]
    if prompt in chat:
        return chat[prompt]
    return "Here is a concise, correct answer."


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts, science, tech, chat = build_prompts()

    model = None
    sp = None

    if RUN_MODEL_FOR_REPORT:
        sp = spm.SentencePieceProcessor()
        sp.load(lc.TOKENIZER_PATH)
        vocab = sp.get_piece_size()

        model = lc.FastGPT(vocab, lc.EMBED_DIM, lc.NUM_HEADS, lc.NUM_LAYERS, lc.BLOCK_SIZE).to(device)
        lc.load_base_checkpoint(model, lc.BASE_CKPT, device)
        for p in model.parameters():
            p.requires_grad = False
        lc.inject_lora(model, device)
        if os.path.exists(lc.LORA_ADAPTER):
            lora_sd = torch.load(lc.LORA_ADAPTER, map_location="cpu")
            lc.load_lora_state_dict(model, lora_sd, device)

    fixes = []
    report = []

    for prompt in prompts:
        chosen = chosen_for_prompt(prompt, science, tech, chat)
        fixes.append({"instruction": prompt, "chosen": chosen})

        if RUN_MODEL_FOR_REPORT and len(report) < REPORT_N:
            before = lc.generate(model, sp, device, prompt, history=[])
            report.append((prompt, before, chosen))

    os.makedirs(os.path.dirname(lc.FEEDBACK_SFT_JSONL), exist_ok=True)
    with open(lc.FEEDBACK_SFT_JSONL, "a", encoding="utf-8") as f:
        for row in fixes:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"added={len(fixes)} → {lc.FEEDBACK_SFT_JSONL}")

    if report:
        for prompt, before, after in report:
            print("----")
            print(f"Q: {prompt}")
            print(f"BEFORE: {before}")
            print(f"AFTER: {after}")


if __name__ == "__main__":
    main()
