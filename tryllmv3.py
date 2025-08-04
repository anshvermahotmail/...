# tinyllm.py
"""
TinyLLM: A minimalist LLM-style chatbot with versioned behaviors (v1, v2, v3, v4)

Supported Versions:
-------------------
1. v1: Pattern-based retrieval from fact file
   - Syntax: v1: where are you?
   - Uses exact or partial match from text lines

2. v2: Vector dot-product matching using tokenized word vectors
   - Syntax: v2: <question>
   - Uses simple average of one-hot vectors and cosine similarity

3. v3: Adds fuzzy fallback — if similarity is below threshold, responds with a default guess
   - Syntax: v3: <question>
   - Threshold configurable (default 0.4)
   - If no good match, fallback to 'Sorry, I don't know.'

4. v4: Auto-learning from past conversations — logs interactions as new Q/A for future
   - Syntax: v4: <question>
   - If no match, adds the Q to log file with placeholder A for manual labeling

5. Default version control:
   - Use: default:v2 or default:v1 to switch global mode

6. Learning syntax:
   - Syntax: learn: q: <question> a: <answer>
   - Adds to both text facts and vector base

7. Help:
   - Syntax: help:
   - Shows full instructions with sample commands

"""

import os
import time
import threading
from datetime import datetime

# Globals
FACT_FILE = "facts.txt"
VECTOR_FILE = "vector_model.txt"
LEARN_LOG = "learn_log.txt"
DEFAULT_VERSION = "v1"
SIM_THRESHOLD = 0.4  # Used in v3
VECTOR_DIM = 10      # Default vector dimension

# --- Utils ---
def timestamp():
    return datetime.now().strftime("%H:%M:%S")

def tokenize(text):
    return [w.lower() for w in text.replace('?', '').replace('.', '').split() if w.lower() not in {'is', 'a', 'an', 'the', 'to', 'and', 'of', 'in'}]

def embed(tokens, dim=VECTOR_DIM):
    vec = [0] * dim
    for t in tokens:
        h = hash(t) % dim
        vec[h] += 1
    return vec

def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def cosine_sim(v1, v2):
    dot_prod = dot(v1, v2)
    mag1 = sum(x*x for x in v1) ** 0.5
    mag2 = sum(y*y for y in v2) ** 0.5
    return dot_prod / (mag1 * mag2 + 1e-6)

# --- File Initialization ---
if not os.path.exists(FACT_FILE):
    with open(FACT_FILE, 'w') as f:
        f.write("q: where are you? a: i am in pune\n")
        f.write("q: what color does ankur like? a: white\n")
        f.write("q: what vehicle does ankur drive? a: electric cycle\n")

# --- Core Functions ---
def learn_entry(q, a):
    with open(FACT_FILE, 'a') as f:
        f.write(f"q: {q} a: {a}\n")
    print(f"[{timestamp()}] [LEARN] Stored: '{q}' → '{a}'")

def match_v1(query):
    with open(FACT_FILE) as f:
        for line in f:
            if line.startswith("q: ") and query.lower() in line.lower():
                return line.split(" a: ")[-1].strip()
    return "I don't know."

def match_v2(query):
    q_tok = tokenize(query)
    q_vec = embed(q_tok)
    best, best_score = None, 0
    with open(FACT_FILE) as f:
        for line in f:
            if line.startswith("q: "):
                q_line = line.split(" a: ")[0][3:].strip()
                a_line = line.split(" a: ")[-1].strip()
                sim = cosine_sim(embed(tokenize(q_line)), q_vec)
                if sim > best_score:
                    best_score = sim
                    best = a_line
    return best if best else "I don't know."

def match_v3(query):
    q_tok = tokenize(query)
    q_vec = embed(q_tok)
    best, best_score = None, 0
    with open(FACT_FILE) as f:
        for line in f:
            if line.startswith("q: "):
                q_line = line.split(" a: ")[0][3:].strip()
                a_line = line.split(" a: ")[-1].strip()
                sim = cosine_sim(embed(tokenize(q_line)), q_vec)
                if sim > best_score:
                    best_score = sim
                    best = a_line
    if best_score >= SIM_THRESHOLD:
        return best
    else:
        return "Sorry, I don't know. [fallback]"

def match_v4(query):
    q_tok = tokenize(query)
    q_vec = embed(q_tok)
    best, best_score = None, 0
    with open(FACT_FILE) as f:
        for line in f:
            if line.startswith("q: "):
                q_line = line.split(" a: ")[0][3:].strip()
                a_line = line.split(" a: ")[-1].strip()
                sim = cosine_sim(embed(tokenize(q_line)), q_vec)
                if sim > best_score:
                    best_score = sim
                    best = a_line
    if best_score >= SIM_THRESHOLD:
        return best
    else:
        with open(LEARN_LOG, 'a') as f:
            f.write(f"q: {query} a: TBD\n")
        return "Let me get back to you on that. [logged for training]"

def show_help():
    print("\n[HELP: Supported Commands]\n")
    print("- default:v1 or default:v2 or v3 or v4 → sets default mode")
    print("- v1: <msg> → Pattern match")
    print("- v2: <msg> → Dot-product similarity")
    print("- v3: <msg> → Dot-product + Fallback")
    print("- v4: <msg> → Dot-product + Logging if not found")
    print("- learn: q: <question> a: <answer> → Add Q/A pair to memory")
    print("- help: → Shows this help")
    print("\n")

# --- Chat Loop ---
def chat():
    global DEFAULT_VERSION
    print(f"[{timestamp()}] TinyLLM Ready. Type 'help:' for commands.")
    while True:
        msg = input("You: ").strip()
        if msg.lower().startswith("default:v"):
            DEFAULT_VERSION = msg.split(":")[1].strip()
            print(f"[{timestamp()}] [INFO] Default version set to {DEFAULT_VERSION}")
        elif msg.lower().startswith("learn:"):
            parts = msg[6:].split(" a: ")
            if len(parts) == 2:
                q = parts[0].replace("q:", "").strip()
                a = parts[1].strip()
                learn_entry(q, a)
            else:
                print(f"[{timestamp()}] [ERROR] Invalid learn syntax.")
        elif msg.lower().startswith("help:"):
            show_help()
        elif msg.lower().startswith("v1:"):
            print(f"[{timestamp()}] Bot: {match_v1(msg[3:].strip())}")
        elif msg.lower().startswith("v2:"):
            print(f"[{timestamp()}] Bot: {match_v2(msg[3:].strip())}")
        elif msg.lower().startswith("v3:"):
            print(f"[{timestamp()}] Bot: {match_v3(msg[3:].strip())}")
        elif msg.lower().startswith("v4:"):
            print(f"[{timestamp()}] Bot: {match_v4(msg[3:].strip())}")
        else:
            if DEFAULT_VERSION == "v1":
                print(f"[{timestamp()}] Bot: {match_v1(msg)}")
            elif DEFAULT_VERSION == "v2":
                print(f"[{timestamp()}] Bot: {match_v2(msg)}")
            elif DEFAULT_VERSION == "v3":
                print(f"[{timestamp()}] Bot: {match_v3(msg)}")
            elif DEFAULT_VERSION == "v4":
                print(f"[{timestamp()}] Bot: {match_v4(msg)}")

if __name__ == "__main__":
    threading.Thread(target=chat).start()
