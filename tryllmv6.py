#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llmfromscratch.py
-----------------
Educational, minimal "LLM-ish" toy system that:
  • Learns a tiny character-level language model from bios you provide.
  • Stores structured facts (name, age, gender, dob, skills, role).
  • Answers queries like:
      what is the age of ankur?
      what can bob do?
      who knows python?
      what is the gender of sarah?
  • Also supports approximate matches via vector embeddings.
  • Supports multiple reasoning versions (v1 - v7) and personality control.
  • Supports file agent actions (v7)
"""

import sys
import os
import re
import json
import numpy as np
import readline
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

facts = []
embeddings = {}
personalities = {
    "child": lambda x: f"{x} 😊",
    "math_prof": lambda x: f"🧮 Math Mode:\n{x}",
    "bio_prof": lambda x: f"🧬 Biology Mode:\n{x}",
    "chemist": lambda x: f"🧪 Chemistry Mode:\n{x}",
    "normal": lambda x: x
}
default_personality = "normal"
current_personality = ""
model = SentenceTransformer("all-MiniLM-L6-v2")
detailed_log = True
last_query_details = {}


def get_current_personality():
    return personalities.get(current_personality or default_personality, personalities["normal"])

def set_personality(key):
    global current_personality
    if key in personalities:
        current_personality = key
        return f"Personality set to: {key}"
    return f"Unknown personality: {key}"

def add_personality(name, func_text):
    try:
        personalities[name] = eval(func_text)
        return f"Added new personality: {name}"
    except Exception as e:
        return f"Failed to add personality: {e}"

def explain_personality():
    doc = "\n\n== Supported Personalities ==\n"
    for k in personalities:
        doc += f"- {k}\n"
    return doc

def parse_bio(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip() and ':' in l]
    d = {}
    for line in lines:
        k, v = line.split(":", 1)
        d[k.strip().lower()] = v.strip()
    return d

def embed(text):
    return model.encode([text])[0]

def similarity(a, b):
    return cosine_similarity([a], [b])[0][0]

def add_fact(d):
    facts.append(d)
    key = d.get("name")
    if key:
        embeddings[key] = embed(" ".join(d.values()))

def search(query):
    tokens = query.lower().split()
    if "who" in tokens and "knows" in tokens:
        skill = tokens[-1]
        return [f"{f['name']} knows {skill}" for f in facts if skill in f.get("skills", "")]
    if "what" in tokens and "age" in tokens:
        name = tokens[-1]
        f = find_by_name(name)
        return [f"{name}'s age is {f.get('age')}"] if f else ["No data"]
    if "what" in tokens and "gender" in tokens:
        name = tokens[-1]
        f = find_by_name(name)
        return [f"{name}'s gender is {f.get('gender')}"] if f else ["No data"]
    if "what" in tokens and "role" in tokens:
        name = tokens[-1]
        f = find_by_name(name)
        return [f"{name}'s role is {f.get('role')}"] if f else ["No data"]
    return ["Unrecognized query"]

def find_by_name(name):
    return next((f for f in facts if f.get("name", "").lower() == name.lower()), None)

def find_similar(text):
    vec = embed(text)
    scores = [(name, similarity(vec, emb)) for name, emb in embeddings.items()]
    scores.sort(key=lambda x: -x[1])
    return scores[:3]

def handle_agent(text):
    actions = []
    if "create" in text and "file" in text:
        match = re.search(r"create\s+(?:file\s+)?([^\s]+)", text)
        if match:
            filename = match.group(1)
            lines = re.findall(r"(?:write|with text|and write)\s+(.*)", text)
            content = "\n".join(lines)
            Path(filename).write_text(content)
            actions.append(f"📄 File created: {filename}")
            if filename.endswith(".bat") or "run" in text:
                try:
                    output = os.popen(filename).read()
                    actions.append(f"▶️ Output:\n{output}")
                except Exception as e:
                    actions.append(f"❌ Error running file: {e}")
        else:
            actions.append("❓ Could not understand agent command.")
    return "\n".join(actions) if actions else "❓ No action performed."

def handle_query(query):
    global current_personality, default_personality, detailed_log
    version = "v1"
    if query.startswith("v") and query[1].isdigit():
        parts = query.split(" ", 1)
        version = parts[0]
        query = parts[1] if len(parts) > 1 else ""

    if query.strip() in ["help:", "help"]:
        return f"""
== Supported Commands ==

- learn: <bio text> → Add structured person info
- v1-v7: ... → Versioned querying
- script: <python code> → Executes python
- agent: <command> → Creates/executes files
- personality: <name> → Set current tone
- default personality: <name> → Sets default tone
- add personality: <name> = lambda x: ...
- help: → Shows this list
{explain_personality()}"""

    if query.startswith("personality:"):
        key = query.split(":", 1)[1].strip()
        return set_personality(key)
    if query.startswith("default personality:"):
        default_personality = query.split(":", 1)[1].strip()
        return f"Default personality set to: {default_personality}"
    if query.startswith("add personality:"):
        name, func = query[len("add personality:"):].split("=", 1)
        return add_personality(name.strip(), func.strip())

    if query.startswith("learn:"):
        d = parse_bio(query[len("learn:"):])
        add_fact(d)
        return f"Learned: {d}"

    if query.startswith("details:"):
        detailed_log = query.split(":", 1)[1].strip().lower() == "on"
        return f"Detailed logging {'enabled' if detailed_log else 'disabled'}"

    if version == "v1":
        return "\n".join(search(query))
    if version == "v2":
        return "\n".join(search(query)) + "\n(Synonyms coming soon)"
    if version == "v3":
        sims = find_similar(query)
        return "\n".join([f"{name}: {score:.2f}" for name, score in sims])
    if version == "v4":
        sims = find_similar(query)
        detail = "\n".join([f"Compared with: {name} → score={score:.2f}" for name, score in sims])
        return detail + "\nAnswer: " + (sims[0][0] if sims else "None")
    if version == "v5":
        if query.startswith("script:"):
            code = query.split(":", 1)[1]
            try:
                exec_globals = {}
                exec(code, exec_globals)
                return str(exec_globals.get("result", "OK"))
            except Exception as e:
                return f"Script error: {e}"
        return "No script provided"
    if version == "v6":
        sims = find_similar(query)
        personality = get_current_personality()
        name = sims[0][0] if sims else "someone"
        resp = f"Answer: {name}"
        return personality(resp)
    if version == "v7" or query.startswith("agent:"):
        cmd = query.split(":", 1)[1] if ":" in query else query
        result = handle_agent(cmd)
        return get_current_personality()(result)

    return "Unknown version"

if __name__ == "__main__":
    print("Type help: to see supported commands\n")
    while True:
        try:
            q = input(">> ")
            print(handle_query(q))
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)