# tiny_llm.py
"""
Tiny LLM Chatbot with Dynamic Vector Embedding and Self-Sufficient Learning
----------------------------------------------------------------------------
Author: GPT Model (Simulated Double Ph.D. in ML)
Description:
This code is a complete, minimalistic chatbot framework that creates its own language model using facts from a file, performs word embedding with dynamic dimension configuration, learns incrementally via user prompts, and runs without external libraries.

Key Features:
- Self-sufficient (no external library used)
- Tokenization and custom vector embeddings
- Dynamic dimension support
- Incremental learning
- Stopword removal and token logging
- WSIG-style console interaction with logging and timestamps
- Concurrent logging and interaction
- Now supports versioned modes: v1: (default), v2: (dot-product question matching)
"""

import os
import time
import threading
from datetime import datetime

# ----------------------------- CONFIGURATIONS ----------------------------- #
FACT_FILE = "facts.txt"
MODEL_FILE = "model.vec"
EMBED_DIM = 3  # Default dimension (can be overridden)
STOP_WORDS = {'is', 'the', 'and', 'a', 'an', 'in', 'to', 'of', 'had', 'has', 'are', 'you'}
DEFAULT_VERSION = "v1"

# ----------------------------- UTILITIES ----------------------------- #
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ----------------------------- FILE HANDLING ----------------------------- #
def initialize_fact_file():
    if not os.path.exists(FACT_FILE):
        with open(FACT_FILE, 'w') as f:
            f.write("Dr. Ankur Rajan Verma (Ph. D) in AI lives in Pune Hadapsar.\n")
            f.write("Ankur likes white color and drives an electric cycle.\n")
            log("Fact file initialized.")

# ----------------------------- TOKENIZATION ----------------------------- #
def tokenize(text):
    tokens = [word.strip('.,!?').lower() for word in text.split() if word.lower() not in STOP_WORDS]
    log(f"Tokenized: {tokens}")
    return tokens

# ----------------------------- EMBEDDING ENGINE ----------------------------- #
def generate_embeddings(tokens, dim):
    vocab = sorted(set(tokens))
    word_to_vec = {}
    for i, word in enumerate(vocab):
        word_to_vec[word] = [(i + 1) * (j + 1) % 7 for j in range(dim)]
        log(f"Embedding for '{word}': {word_to_vec[word]}")
    return word_to_vec

def vectorize(tokens, dim):
    if not tokens:
        return [0] * dim
    vectors = []
    for i, word in enumerate(tokens):
        vec = [(ord(c) * (j + 1)) % 7 for j, c in enumerate(word[:dim])]
        while len(vec) < dim:
            vec.append(0)
        vectors.append(vec)
    avg_vec = [sum(col) // len(col) for col in zip(*vectors)]
    return avg_vec

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

# ----------------------------- MODEL TRAINING ----------------------------- #
def train_model_from_facts(dim):
    with open(FACT_FILE, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        tokens.extend(tokenize(line))
    word_to_vec = generate_embeddings(tokens, dim)
    with open(MODEL_FILE, 'w') as f:
        for word, vec in word_to_vec.items():
            f.write(f"{word}:{','.join(map(str, vec))}\n")
    log(f"Model trained with {len(word_to_vec)} words in {dim}D space.")

# ----------------------------- LEARNING NEW FACT ----------------------------- #
def learn_new_fact(sentence, dim):
    with open(FACT_FILE, 'a') as f:
        f.write(sentence + "\n")
    train_model_from_facts(dim)

# ----------------------------- RESPONSE ENGINE v1 ----------------------------- #
def respond_v1(query):
    tokens = tokenize(query)
    if not tokens:
        print("Bot: I don't understand.")
        return
    found = False
    with open(FACT_FILE, 'r') as f:
        for line in f:
            if all(tok in line.lower() for tok in tokens):
                print("Bot:", line.strip())
                found = True
    if not found:
        print("Bot: I don't know that. Try teaching me using 'learn:'")

# ----------------------------- RESPONSE ENGINE v2 ----------------------------- #
def respond_v2(query):
    tokens = tokenize(query)
    if not tokens:
        print("Bot: I don't understand.")
        return
    query_vec = vectorize(tokens, EMBED_DIM)
    best_score = 0
    best_response = None

    with open(FACT_FILE, 'r') as f:
        for line in f:
            if line.lower().startswith("q:") and " a:" in line:
                parts = line.strip().split(" a:", 1)
                q_part = parts[0][2:].strip()
                a_part = parts[1].strip()
                q_tokens = tokenize(q_part)
                q_vec = vectorize(q_tokens, EMBED_DIM)
                score = dot_product(query_vec, q_vec)
                if score > best_score:
                    best_score = score
                    best_response = a_part

    if best_response:
        print("Bot:", best_response)
    else:
        print("Bot: I don't know that. Try teaching me using 'learn: q:... a:...' format.")

# ----------------------------- CHATBOT CORE ----------------------------- #
def chatbot():
    global EMBED_DIM, DEFAULT_VERSION
    initialize_fact_file()
    train_model_from_facts(EMBED_DIM)
    log("Chatbot ready. Type your query or 'learn: your_fact_here'. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower().startswith("learn:"):
            new_fact = user_input[len("learn:"):].strip()
            learn_new_fact(new_fact, EMBED_DIM)
            log("Learned new fact.")
        elif user_input.lower().startswith("dimension:"):
            try:
                EMBED_DIM = int(user_input[len("dimension:"):].strip())
                train_model_from_facts(EMBED_DIM)
                log(f"Changed embedding dimension to {EMBED_DIM}.")
            except ValueError:
                log("Invalid dimension.")
        elif user_input.lower().startswith("default:v1"):
            DEFAULT_VERSION = "v1"
            log("Switched default version to v1.")
        elif user_input.lower().startswith("default:v2"):
            DEFAULT_VERSION = "v2"
            log("Switched default version to v2.")
        elif user_input.lower() == "exit":
            log("Goodbye.")
            break
        elif user_input.lower().startswith("v1:"):
            respond_v1(user_input[len("v1:"):].strip())
        elif user_input.lower().startswith("v2:"):
            respond_v2(user_input[len("v2:"):].strip())
        else:
            if DEFAULT_VERSION == "v1":
                respond_v1(user_input)
            else:
                respond_v2(user_input)

# ----------------------------- FACT STRUCTURE SUGGESTION ----------------------------- #
def suggest_fact_structure():
    log("Suggestion: Organize facts in 'q: question a: answer' format for better retrieval in v2 mode.")
    log("Use 'learn: q: where are you? a: i am in pune' to teach specific pairs.")

# ----------------------------- MAIN ----------------------------- #
if __name__ == "__main__":
    threading.Thread(target=suggest_fact_structure).start()
    chatbot()
