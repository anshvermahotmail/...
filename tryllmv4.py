#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_extended.py
A minimal toy LLM system with multi-version support, embedding, command parsing, explainability,
personality-driven responses, shell scripting, and agent actions.
"""

import numpy as np
import os
import re
import subprocess
from collections import defaultdict
import math

# ------------------------------
# Global Knowledge Store
# ------------------------------
facts = []
vectors = []
questions = []
answers = []
personalities = {
    "neutral": lambda x: x,
    "child": lambda x: f"Hmm okay! So, {x} 🤓✨",
    "professor": lambda x: f"Let me elaborate mathematically: {x}",
    "math": lambda x: f"In mathematical terms: {x}",
    "biology": lambda x: f"In biology, we'd say: {x}",
    "chemistry": lambda x: f"Chemically speaking: {x}"
}
default_personality = "neutral"
default_version = "v6"
details_on = True
last_response = ""

# ------------------------------
# Tokenization & Vectorization
# ------------------------------
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def embed(text):
    tokens = tokenize(text)
    vec = np.zeros(300)
    for t in tokens:
        np.random.seed(abs(hash(t)) % (10 ** 8))
        vec += np.random.rand(300)
    return vec / (np.linalg.norm(vec) + 1e-8)

# ------------------------------
# Similarity Functions
# ------------------------------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

def dot_similarity(v1, v2):
    return np.dot(v1, v2)

# ------------------------------
# Personality System
# ------------------------------
def apply_personality(text):
    func = personalities.get(default_personality, personalities["neutral"])
    return func(text)

def set_personality(name):
    global default_personality
    if name in personalities:
        default_personality = name
        return f"Personality set to '{name}'."
    return f"Unknown personality '{name}'. Use 'personality:' to list."

def add_personality(name, style):
    personalities[name] = lambda x: style.replace("{x}", x)
    return f"Personality '{name}' added."

# ------------------------------
# Answering Logic per Version
# ------------------------------
def answer_v1(q):
    if q in questions:
        return answers[questions.index(q)]
    return "I don't know that."

def answer_v2(q):
    vec = embed(q)
    sims = [dot_similarity(vec, v) for v in vectors]
    idx = np.argmax(sims)
    return answers[idx] if sims[idx] > 0 else "Not found."

def answer_v3(q):
    vec = embed(q)
    sims = [cosine_similarity(vec, v) for v in vectors]
    idx = np.argmax(sims)
    return answers[idx] if sims[idx] > 0.5 else "No close match found."

def answer_v4(q):
    vec = embed(q)
    tokens = set(tokenize(q))
    scored = []
    for i, v in enumerate(vectors):
        score = cosine_similarity(vec, v)
        boost = 0.1 * len(tokens & set(tokenize(questions[i])))
        scored.append((score + boost, i))
    idx = max(scored)[1]
    return answers[idx]

def answer_v5(q):
    global last_response
    vec = embed(q)
    sims = [cosine_similarity(vec, v) for v in vectors]
    idx = np.argmax(sims)
    last_response = f"Query: {q}\nVec: {vec[:5]}\nBest Match: {questions[idx]}\nSim: {sims[idx]}"
    return answers[idx]

def answer_v6(q):
    base = answer_v5(q)
    return apply_personality(base)

# ------------------------------
# Core Query Handler
# ------------------------------
def handle_query(q, version=None):
    if not version:
        version = default_version
    method = {
        "v1": answer_v1,
        "v2": answer_v2,
        "v3": answer_v3,
        "v4": answer_v4,
        "v5": answer_v5,
        "v6": answer_v6
    }.get(version, answer_v6)
    response = method(q)
    if details_on:
        response += f"\n\n[version: {version}, personality: {default_personality}]"
    return response

# ------------------------------
# Command Processor
# ------------------------------
def process(command):
    global default_version, details_on
    c = command.strip()

    if c.lower().startswith("learn:"):
        parts = c[6:].split("=")
        if len(parts) == 2:
            q, a = parts[0].strip(), parts[1].strip()
            questions.append(q)
            answers.append(a)
            vectors.append(embed(q))
            return f"Learned: '{q}' ➜ '{a}'"
        return "Syntax: learn: question = answer"

    elif c.lower().startswith("default:"):
        v = c.split(":")[1].strip()
        if v in {"v1", "v2", "v3", "v4", "v5", "v6"}:
            default_version = v
            return f"Default version set to {v}"
        return f"Supported: v1–v6"

    elif c.lower().startswith("help"):
        return """📘 Supported Commands:
- learn: question = answer
- help or help:
- default: vX       → set default version
- personality:      → list personalities
- personality: <p>  → set active personality
- personality: add <p> = style text using {x}
- details: yes/no   → toggle detailed info
- explain:          → show explanation of last response
- script: <cmd>     → run OS command
- agent: <task>     → multi-step action (like file create/run)
"""

    elif c.lower().startswith("details:"):
        val = c.split(":")[1].strip().lower()
        details_on = (val == "yes")
        return f"Details set to {details_on}"

    elif c.lower().startswith("explain:"):
        return last_response if last_response else "Nothing to explain yet."

    elif c.lower().startswith("personality:"):
        args = c[12:].strip()
        if args == "":
            return "Available: " + ", ".join(personalities.keys())
        if args.startswith("add"):
            name, style = args[4:].split("=", 1)
            return add_personality(name.strip(), style.strip())
        return set_personality(args)

    elif c.lower().startswith("script:"):
        cmd = c[7:].strip()
        try:
            out = subprocess.check_output(cmd, shell=True, text=True)
            return f"📂 Output:\n{out}"
        except Exception as e:
            return f"❌ Error: {e}"

    elif c.lower().startswith("agent:"):
        task = c[6:].strip()
        if "create a file" in task:
            match = re.search(r"file ([\w.\-]+)", task)
            if match:
                fname = match.group(1)
                content = re.findall(r"(write|with lines?) (.+)", task)
                lines = content[0][1] if content else "hello world"
                if not os.path.exists(fname):
                    with open(fname, "w") as f:
                        f.write(lines + "\n")
                if fname.endswith(".bat"):
                    return process(f"script: {fname}")
                return f"✅ Created file '{fname}' with:\n{lines}"
        return "Unknown agent action."

    else:
        return handle_query(c)

# ------------------------------
# REPL
# ------------------------------
if __name__ == "__main__":
    print("🤖 LLM Toy v6 - Type 'help' for commands.")
    while True:
        try:
            user_input = input("\n>> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            print(process(user_input))
        except KeyboardInterrupt:
            print("\nBye.")
            break
