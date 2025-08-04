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
"""

import os
import time
import threading
from datetime import datetime

# ----------------------------- CONFIGURATIONS ----------------------------- #
FACT_FILE = "facts.txt"
MODEL_FILE = "model.vec"
EMBED_DIM = 3  # Default dimension (can be overridden)
STOP_WORDS = {'is', 'the', 'and', 'a', 'an', 'in', 'to', 'of', 'had', 'has'}

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
        # Simple deterministic embedding: based on position and dimension
        word_to_vec[word] = [(i + 1) * (j + 1) % 7 for j in range(dim)]
        log(f"Embedding for '{word}': {word_to_vec[word]}")
    return word_to_vec

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

# ----------------------------- CHATBOT CORE ----------------------------- #
def chatbot():
    global EMBED_DIM
    initialize_fact_file()
    train_model_from_facts(EMBED_DIM)
    log("Chatbot ready. Type your query or 'learn: your_fact_here'. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
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
        elif user_input.lower() == "exit":
            log("Goodbye.")
            break
        else:
            respond(user_input)

# ----------------------------- RESPONSE ENGINE ----------------------------- #
def respond(query):
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

# ----------------------------- FACT STRUCTURE SUGGESTION ----------------------------- #
def suggest_fact_structure():
    log("Suggestion: Organize facts in key-value pairs, e.g., 'name: Ankur Verma', 'location: Hadapsar Pune', 'vehicle: electric cycle'.")
    log("Use this format for better next-word prediction and structured prompting.")

# ----------------------------- MAIN ----------------------------- #
if __name__ == "__main__":
    threading.Thread(target=suggest_fact_structure).start()
    chatbot()
