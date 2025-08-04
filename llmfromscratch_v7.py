#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llmfromscratch_enhanced.py
-----------------
Educational minimal "LLM-ish" toy system combining structured fact lookup
with a tiny character-level LLM for generative augmentation and fallback.
"""

import math
import random
import re
import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np

# ------------------ RNG SEEDS ------------------
random.seed(42)
np.random.seed(42)

# ------------------ CONFIG ------------------
BLOCK_SIZE = 16  # How many previous characters model sees to predict next
EMBED_SIZE = 32  # Embedding dimension
USE_ATTENTION = True  # If False, model skips attention and mean-pools embeddings
LEARNING_RATE = 1e-2
EPOCHS = 1000  # Number of update steps
PRINT_EVERY = 100  # Print loss frequency
TOPK_GENERATE = 5  # Top-k sampling for generative flavor text

# ------------------ 1. TOKENIZER ------------------
def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    ids = [stoi[ch] for ch in text if ch in stoi]
    return ids

def decode(indices: List[int], itos: Dict[int, str]) -> str:
    return ''.join(itos[i] for i in indices if i in itos)

# ------------------ 2. POSITIONAL ENCODING ------------------
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pos_enc = np.zeros((seq_len, d_model), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** ((2 * i) / d_model))
            pos_enc[pos, i] = math.sin(angle)
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = math.cos(angle)
    return pos_enc

# ------------------ 3. CORE MATH UTILITIES ------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def cross_entropy_from_probs(probs: np.ndarray, target_idx: int) -> float:
    target_idx = int(target_idx)
    if target_idx < 0 or target_idx >= probs.size:
        target_idx = min(max(target_idx, 0), probs.size - 1)
    return -math.log(float(probs[target_idx]) + 1e-8)

# ------------------ 4. MODEL INITIALIZATION ------------------
def init_weights(vocab_size: int, embed_size: int = EMBED_SIZE) -> Dict[str, np.ndarray]:
    scale = 0.02
    params = {
        "W_embed": np.random.randn(vocab_size, embed_size) * scale,
        "W_q": np.random.randn(embed_size, embed_size) * scale,
        "W_k": np.random.randn(embed_size, embed_size) * scale,
        "W_v": np.random.randn(embed_size, embed_size) * scale,
        "W_o": np.random.randn(embed_size, embed_size) * scale,
        "W_fc": np.random.randn(embed_size, vocab_size) * scale,
    }
    return params

# ------------------ 5. FORWARD PASS ------------------
def forward_block(x_ids: List[int], params: Dict[str, np.ndarray], pos_enc: np.ndarray, use_attention=True) -> Tuple[np.ndarray,np.ndarray]:
    W_embed = params["W_embed"]
    W_q = params["W_q"]
    W_k = params["W_k"]
    W_v = params["W_v"]
    W_o = params["W_o"]
    W_fc = params["W_fc"]

    x_embed = W_embed[x_ids]
    T = x_embed.shape[0]
    x_embed = x_embed + pos_enc[:T, :]

    if use_attention:
        Q = x_embed @ W_q
        K = x_embed @ W_k
        V = x_embed @ W_v
        scores = (Q @ K.T) / math.sqrt(Q.shape[-1])
        mask = np.triu(np.ones((T,T), dtype=bool), k=1)
        scores = np.where(mask, -1e10, scores)
        attn = softmax(scores, axis=-1)
        context = attn @ V
        h = context @ W_o
    else:
        h_mean = x_embed.mean(axis=0, keepdims=True)
        h = np.repeat(h_mean, T, axis=0)

    h_last = h[-1]
    logits = h_last @ W_fc
    probs = softmax(logits, axis=-1)
    return probs, h_last

# ------------------ 6. TRAINING ------------------
def train_model(data_ids: List[int], params: Dict[str, np.ndarray], pos_enc: np.ndarray,
                epochs: int=EPOCHS, lr: float=LEARNING_RATE, use_attention: bool=USE_ATTENTION,
                print_every: int=PRINT_EVERY) -> None:
    W_fc = params["W_fc"]
    vocab_size = W_fc.shape[1]
    N = len(data_ids)
    if N <= BLOCK_SIZE + 1:
        print("⚠️ Not enough data to train (need > BLOCK_SIZE + 1 tokens).")
        return

    for epoch in range(epochs):
        i = random.randint(0, N - BLOCK_SIZE - 1)
        x = data_ids[i : i + BLOCK_SIZE]
        y = data_ids[i + 1 : i + BLOCK_SIZE + 1]

        probs, h_last = forward_block(x, params, pos_enc, use_attention)
        target_idx = y[-1]
        if target_idx >= vocab_size:
            target_idx = vocab_size - 1

        loss = cross_entropy_from_probs(probs, target_idx)
        dlogits = probs.copy()
        dlogits[target_idx] -= 1.0
        grad_W_fc = np.outer(h_last, dlogits)
        W_fc -= lr * grad_W_fc

        if epoch % print_every == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

# ------------------ 7. GENERATION ------------------
def sample_next_char(probs: np.ndarray, topk: int=TOPK_GENERATE) -> int:
    if topk is not None and topk > 0:
        top_idx = np.argpartition(probs, -topk)[-topk:]
        top_probs = probs[top_idx]
        top_probs = top_probs / top_probs.sum()
        return int(np.random.choice(top_idx, p=top_probs))
    else:
        return int(np.random.choice(len(probs), p=probs))

def generate_text(prompt: str, params: Dict[str, np.ndarray], stoi: Dict[str,int], itos: Dict[int,str],
                  pos_enc: np.ndarray, max_new_tokens: int=128, use_attention: bool=USE_ATTENTION) -> str:
    context = encode(prompt, stoi)
    if not context:
        context = [0]
    for _ in range(max_new_tokens):
        x = context[-BLOCK_SIZE:]
        probs, _ = forward_block(x, params, pos_enc, use_attention)
        nxt = sample_next_char(probs)
        context.append(nxt)
    return decode(context, itos)

# ------------------ 8. STRUCTURED KNOWLEDGE PARSING ------------------
BIO_PAT = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Z][A-Za-z0-9_ ]+?)\s+is\s+
    (?:a\s+)?(?P<age>\d{1,3})[- ]?year[- ]?old\s+
    (?P<gender>male|female|m|f|man|woman)?
    .*?
    (?:born\s+on\s+(?P<dob>\d{4}-\d{2}-\d{2}))?
    .*$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

ROLE_PAT = re.compile(r"works\s+as\s+a?n?\s+([^\.]+)", re.IGNORECASE)
SKILL_PAT = re.compile(r"(skilled\s+in|expert\s+in|knows)\s+([^\.]+)", re.IGNORECASE)

def parse_bio_line(line: str) -> Dict[str, Any]:
    info = {"name": None, "age": None, "gender": None, "dob": None,
            "skills": [], "role": None, "raw": line.strip()}
    m = BIO_PAT.search(line)
    if m:
        info["name"] = m.group("name").strip().split()[0]
        info["age"] = safe_int(m.group("age"))
        info["gender"] = norm_gender(m.group("gender"))
        info["dob"] = m.group("dob")
    m2 = ROLE_PAT.search(line)
    if m2:
        info["role"] = m2.group(1).strip()
    m3 = SKILL_PAT.search(line)
    if m3:
        skills_str = m3.group(2)
        info["skills"] = split_skills(skills_str)
    if not info["name"]:
        tokens = line.strip().split()
        if tokens:
            info["name"] = tokens[0]
    return info

def safe_int(x): 
    try: return int(x)
    except Exception: return None

def norm_gender(g): 
    if not g: return None
    g = g.lower()
    if g in ("m", "male", "man"): return "male"
    if g in ("f", "female", "woman"): return "female"
    return g

def split_skills(s: str) -> List[str]:
    s = s.replace(" and ", ", ")
    parts = [p.strip(" .") for p in s.split(",")]
    return [p for p in parts if p]

def build_people_knowledge(text: str) -> Dict[str, Dict[str, Any]]:
    people = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        info = parse_bio_line(line)
        if info["name"]:
            people[info["name"].lower()] = info
    return people

# ------------------ 9. ANSWER QUERIES - ENHANCED ------------------
def answer_about(name: str,
                 people_db: Dict[str, Dict[str, Any]],
                 params: Dict[str, np.ndarray],
                 stoi: Dict[str, int],
                 itos: Dict[int, str],
                 pos_enc: np.ndarray,
                 use_attention: bool = USE_ATTENTION,
                 use_model_flavor: bool = True
                 ) -> Tuple[str, List[str]]:
    logs = []
    key = name.lower()
    logs.append(f"Lookup structured info for '{name}' (key='{key}').")

    if key in people_db:
        p = people_db[key]
        parts = [f"{p['name']}"]
        if p["age"] is not None:
            parts.append(f"is {p['age']} years old")
        if p["gender"]:
            parts.append(p["gender"])
        if p["dob"]:
            parts.append(f"(born {p['dob']})")
        if p["role"]:
            parts.append(f"and works as {p['role']}")
        if p["skills"]:
            parts.append("with skills in " + ", ".join(p["skills"]))

        fact_resp = " ".join(parts) + "."
        logs.append("Structured fact found:")
        logs.append(fact_resp)

        if use_model_flavor:
            logs.append("Generating model flavor text continuation...")
            prompt = f" {name} "
            gen = generate_text(prompt, params, stoi, itos, pos_enc,
                                max_new_tokens=40, use_attention=use_attention)
            logs.append(f"Model generated continuation: {gen.strip()}")
            full_resp = fact_resp + "\n(model continuation) " + gen.strip()
        else:
            full_resp = fact_resp

        logs.append("Final combined response assembled.")
        return full_resp, logs

    else:
        logs.append(f"No structured info found for '{name}'. Using pure generative fallback.")
        prompt = f"Tell me about {name}"
        gen = generate_text(prompt, params, stoi, itos, pos_enc,
                            max_new_tokens=80, use_attention=use_attention)
        logs.append(f"Model generation output: {gen.strip()}")
        fallback_resp = f"I don't have structured info on {name}. Here's what the tiny model says:\n{gen}"
        logs.append("Final fallback response assembled.")
        return fallback_resp, logs

# ------------------ 10. ADD / UPDATE PERSON ------------------
def add_person(bio_line: str,
               people_db: Dict[str, Dict[str, Any]],
               full_text: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    full_text = full_text.rstrip() + "\n" + bio_line.strip() + "\n"
    info = parse_bio_line(bio_line)
    if info["name"]:
        people_db[info["name"].lower()] = info
    return full_text, people_db

# ------------------ 11. SAVE / LOAD WEIGHTS ------------------
def save_weights(path: str, params: Dict[str, np.ndarray]) -> None:
    data = {k: v.tolist() for k, v in params.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_weights(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    return params

# ------------------ 12. DEMO MAIN ------------------
if __name__ == "__main__":
    initial_data = """\
John is a 28-year-old male. He is skilled in Python, Java, and Kubernetes. He works as a DevOps Engineer.
Aisha is a 25-year-old female. She is an expert in Data Science, ML, and SQL. She works as a Data Analyst.
"""

    # Parse structured knowledge base
    people_db = build_people_knowledge(initial_data)

    # Build vocab and encode data
    stoi, itos = build_vocab(initial_data)
    vocab_size = len(stoi)
    data_ids = encode(initial_data, stoi)

    if len(data_ids) < BLOCK_SIZE + 2:
        print("⚠️ Not enough data to train meaningfully.")
        exit(0)

    # Initialize model parameters and positional encoding
    params = init_weights(vocab_size, EMBED_SIZE)
    pos_enc = positional_encoding(BLOCK_SIZE, EMBED_SIZE)

    # Train the model
    print("=== Training on initial data ===")
    train_model(data_ids, params, pos_enc, epochs=EPOCHS,
                lr=LEARNING_RATE, use_attention=USE_ATTENTION,
                print_every=PRINT_EVERY)

    # Queries with logs
    for query_name in ["John", "Aisha", "Ravi"]:
        print(f"\n--- Query: {query_name} ---")
        response, debug_logs = answer_about(query_name, people_db, params, stoi, itos, pos_enc, use_attention=USE_ATTENTION, use_model_flavor=True)
        print(response)
        print("\n[Debug logs]")
        for log in debug_logs:
            print("  " + log)

    # Adding new person
    new_bio = "Michael is a 30-year-old male. He works in Cloud Security and knows AWS and GCP."
    print("\n=== Adding new person: Michael ===")
    initial_data, people_db = add_person(new_bio, people_db, initial_data)

    # Rebuild vocab and data ids
    stoi, itos = build_vocab(initial_data)
    vocab_size = len(stoi)
    data_ids = encode(initial_data, stoi)

    # Re-initialize & retrain
    params = init_weights(vocab_size, EMBED_SIZE)
    pos_enc = positional_encoding(BLOCK_SIZE, EMBED_SIZE)

    print("=== Retraining with Michael included ===")
    train_model(data_ids, params, pos_enc, epochs=EPOCHS,
                lr=LEARNING_RATE, use_attention=USE_ATTENTION,
                print_every=PRINT_EVERY)

    # Query new person Michael
    print("\n--- Query: Michael ---")
    response, debug_logs = answer_about("Michael", people_db, params, stoi, itos, pos_enc, use_attention=USE_ATTENTION, use_model_flavor=True)
    print(response)
    print("\n[Debug logs]")
    for log in debug_logs:
        print("  " + log)

    # Optional save
    weights_path = "llm_toy_weights.json"
    save_weights(weights_path, params)
    print(f"\nWeights saved to {os.path.abspath(weights_path)}")
