import os
import re
import json
import time
import math
import glob
import shutil
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Core config
DATA_FILE = "facts.json"
VERSION = "v7"
EMBED_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)

# Runtime states
facts = []
vectors = []
default_personality = "normal"
personalities = {
    "normal": lambda x: x,
    "child": lambda x: f"{x} 😊",
    "boy": lambda x: f"Yo! {x}",
    "professor": lambda x: f"Let me explain that precisely:\n\n{x}",
    "maths": lambda x: f"🧮 Math Explanation:\n{x}",
    "biology": lambda x: f"🧬 Bio Log:\n{x}",
    "chemistry": lambda x: f"🧪 Chem Talk:\n{x}",
}

# Detailed explanation toggle
detailed_log = True
last_query_details = {}

# Save/load fact DB
def save_facts():
    with open(DATA_FILE, "w") as f:
        json.dump(facts, f, indent=2)

def load_facts():
    global facts, vectors
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            facts = json.load(f)
        vectors = model.encode([f["text"] for f in facts])

# Embed a new fact
def learn_fact(text):
    embedding = model.encode([text])[0]
    facts.append({"text": text, "embedding": embedding.tolist()})
    save_facts()
    return f"✅ Learned: {text}"

# Match input to closest fact
def match_fact(query):
    query_vector = model.encode([query])
    scores = cosine_similarity(query_vector, vectors)[0]
    top = np.argmax(scores)
    similarity = float(scores[top])
    response = facts[top]["text"]
    if detailed_log:
        print(f"🔍 Match score: {similarity:.4f}")
        print(f"🔗 Closest: {response}")
    global last_query_details
    last_query_details = {
        "query": query,
        "similarity": similarity,
        "match": response
    }
    return response

# Normalization
def clean_input(text):
    return re.sub(r"\s*([?.!,])", r"\1", text.strip())

# Handle agent actions
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

# Handle query dispatch
def handle_query(user_input):
    global detailed_log, VERSION, default_personality

    text = clean_input(user_input)
    if ":" in text:
        cmd, value = text.split(":", 1)
        cmd, value = cmd.strip().lower(), value.strip()
    else:
        cmd, value = "", text

    cmd_map = {"v1": 1, "v2": 2, "v3": 3, "v4": 4, "v5": 5, "v6": 6, "v7": 7}
    version = cmd if cmd in cmd_map else VERSION

    # Help
    if cmd == "help":
        return (
            "📘 Help - Commands Supported:\n"
            "- `learn: fact` → Save fact\n"
            "- `reset:` → Clear facts\n"
            "- `explain:` → Show last match details\n"
            "- `default:` → Show/set default version\n"
            "- `personality: name` → Set personality (e.g., child, boy, professor)\n"
            "- `agent: command` or `v7:` → Perform file actions\n"
            "- `details: on/off` → Enable verbose\n"
            "- `v1:` to `v7:` → Choose version\n"
            "- `help:` → Show this help"
        )

    # Details logging
    if cmd == "details":
        detailed_log = value.lower() != "off"
        return f"🔧 Detailed logging {'enabled' if detailed_log else 'disabled'}."

    # Default version
    if cmd == "default":
        if value in cmd_map:
            VERSION = value
            return f"✅ Default version set to {value}"
        return f"ℹ️ Current default version: {VERSION}"

    # Personality handling
    if cmd == "personality":
        if value not in personalities:
            return f"⚠️ Unknown personality. Try: {', '.join(personalities)}"
        default_personality = value
        return f"🎭 Personality set to: {value}"

    # Learn
    if cmd == "learn":
        return learn_fact(value)

    # Reset
    if cmd == "reset":
        facts.clear()
        save_facts()
        return "🗑️ All facts deleted."

    # Explain
    if cmd == "explain":
        if not last_query_details:
            return "ℹ️ No query yet to explain."
        return json.dumps(last_query_details, indent=2)

    # Agent
    if cmd == "agent" or version == "v7":
        result = handle_agent(value)
        return personalities[default_personality](result)

    # Fallback to embedding match
    if not facts:
        return "⚠️ No facts available. Use `learn:` to add."

    response = match_fact(value)
    return personalities[default_personality](response)

# CLI interface
def cli_loop():
    load_facts()
    print(f"🧠 Tiny LLM Chatbot - Default {VERSION} | Type `help:`")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            response = handle_query(user_input)
            print(f"🤖: {response}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    cli_loop()
