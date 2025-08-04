#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llmfromscratch.py
-----------------
Educational, minimal "LLM-ish" toy system that:
  • Learns a tiny character-level language model from bios you provide.
  • Stores structured facts (name, age, gender, dob, skills, role).
  • Answers queries like: "Tell me about John" using structured memory,
    optionally augmented by a tiny learned next-character model.
  • Lets you add new people, retrain, save/load weights.

This is NOT a real LLM; it's a teaching scaffold to understand core ideas:
tokenization, embeddings, (toy) attention, softmax, cross-entropy, training.
"""

import math
import random
import re
import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np


# ---------------------------------------------------------------------------
# RNG SEEDS (helps reproducibility)
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# CONFIG (feel free to tweak)
# ---------------------------------------------------------------------------
BLOCK_SIZE   = 16   # how many previous characters model sees to predict next
EMBED_SIZE   = 32   # embedding dimension
USE_ATTENTION = True  # if False, model skips attention and mean-pools embeddings
LEARNING_RATE = 1e-2
EPOCHS        = 1000    # number of update steps
PRINT_EVERY   = 100     # print loss frequency
TOPK_GENERATE = 5       # top-k sampling for generative flavor text


# ---------------------------------------------------------------------------
# 1. TOKENIZER (character-level)
# ---------------------------------------------------------------------------
def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build string->index (stoi) and index->string (itos) maps from text."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    """Convert text to list of token IDs. Unknown chars are skipped."""
    ids = []
    for ch in text:
        if ch in stoi:
            ids.append(stoi[ch])
        # else silently skip (or append an <unk> index if you add one)
    return ids


def decode(indices: List[int], itos: Dict[int, str]) -> str:
    """Convert list of IDs back to string."""
    return ''.join(itos[i] for i in indices if i in itos)


# ---------------------------------------------------------------------------
# 2. POSITIONAL ENCODING (classic sin/cos — optional; we just precompute)
# ---------------------------------------------------------------------------
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Create sinusoidal positional encodings.
    pos_enc[pos, i] = sin(pos / (10000^(2i/d_model))) for even i
                    = cos(pos / (10000^(2i/d_model))) for odd  i
    Helps model know where tokens occur in a sequence.
    """
    pos_enc = np.zeros((seq_len, d_model), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** ( (2 * i) / d_model ))
            pos_enc[pos, i] = math.sin(angle)
            if i + 1 < d_model:
                pos_enc[pos, i+1] = math.cos(angle)
    return pos_enc


# ---------------------------------------------------------------------------
# 3. CORE MATH UTILITIES
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax converts raw scores (logits) to probabilities:
        softmax(z_i) = exp(z_i - max_z) / sum_j exp(z_j - max_z)
    We subtract max for numerical stability.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def cross_entropy_from_probs(probs: np.ndarray, target_idx: int) -> float:
    """
    Cross-entropy loss for 1 example:
        L = -log(prob[target])
    Add epsilon to avoid log(0).
    """
    target_idx = int(target_idx)
    if target_idx < 0 or target_idx >= probs.size:
        # clamp for safety — shouldn't happen if data + vocab consistent
        target_idx = min(max(target_idx, 0), probs.size - 1)
    return -math.log(float(probs[target_idx]) + 1e-8)


# ---------------------------------------------------------------------------
# 4. MODEL INITIALIZATION
# ---------------------------------------------------------------------------
def init_weights(vocab_size: int,
                 embed_size: int = EMBED_SIZE) -> Dict[str, np.ndarray]:
    """
    Initialize learnable parameters.
    Shapes:
      W_embed : (vocab_size, embed_size)
      W_q     : (embed_size, embed_size)
      W_k     : (embed_size, embed_size)
      W_v     : (embed_size, embed_size)
      W_o     : (embed_size, embed_size)   # attention output proj
      W_fc    : (embed_size, vocab_size)   # to logits
    """
    scale = 0.02  # small init
    params = {
        "W_embed": np.random.randn(vocab_size, embed_size) * scale,
        "W_q":     np.random.randn(embed_size, embed_size) * scale,
        "W_k":     np.random.randn(embed_size, embed_size) * scale,
        "W_v":     np.random.randn(embed_size, embed_size) * scale,
        "W_o":     np.random.randn(embed_size, embed_size) * scale,
        "W_fc":    np.random.randn(embed_size, vocab_size) * scale,
    }
    return params


# ---------------------------------------------------------------------------
# 5. FORWARD PASS (single block, single head)
# ---------------------------------------------------------------------------
def forward_block(x_ids: List[int],
                  params: Dict[str, np.ndarray],
                  pos_enc: np.ndarray,
                  use_attention: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward pass for a single sequence (length <= BLOCK_SIZE).

    Inputs:
      x_ids  : list[int], token IDs
      params : dict of weight matrices
      pos_enc: (BLOCK_SIZE, EMBED_SIZE) precomputed
      use_attention: if False, do mean-pool instead of attention

    Returns:
      probs  : (vocab_size,) probability distribution for NEXT token
      h_last : (embed_size,) last-step hidden rep (for gradient demos)
    """
    W_embed = params["W_embed"]
    W_q = params["W_q"]
    W_k = params["W_k"]
    W_v = params["W_v"]
    W_o = params["W_o"]
    W_fc = params["W_fc"]

    # Embed tokens
    x_embed = W_embed[x_ids]  # shape (T, EMBED_SIZE)
    T = x_embed.shape[0]

    # Add positional encoding (truncate to T)
    x_embed = x_embed + pos_enc[:T, :]

    if use_attention:
        # Single-head self-attention (causal)
        Q = x_embed @ W_q  # (T, E)
        K = x_embed @ W_k  # (T, E)
        V = x_embed @ W_v  # (T, E)

        # Compute raw attention scores
        # scores[t, s] = dot(Q[t], K[s])
        scores = (Q @ K.T) / math.sqrt(Q.shape[-1])  # (T, T)

        # Causal mask: positions can only attend to <= current index
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # True above diag
        scores = np.where(mask, -1e10, scores)  # large negative = masked

        attn = softmax(scores, axis=-1)  # (T, T)

        # Weighted sum
        context = attn @ V  # (T, E)

        # Output projection
        h = context @ W_o  # (T, E)

    else:
        # Simpler baseline: just mean-pool the embeddings
        h_mean = x_embed.mean(axis=0, keepdims=True)  # (1, E)
        h = np.repeat(h_mean, T, axis=0)  # pretend we have T steps, all same

    # Use the last time step to predict NEXT token
    h_last = h[-1]  # (E,)

    logits = h_last @ W_fc  # (vocab_size,)
    probs = softmax(logits, axis=-1)  # (vocab_size,)
    return probs, h_last


# ---------------------------------------------------------------------------
# 6. TRAINING (naive SGD on W_fc only, for clarity)
# ---------------------------------------------------------------------------
def train_model(data_ids: List[int],
                params: Dict[str, np.ndarray],
                pos_enc: np.ndarray,
                epochs: int = EPOCHS,
                lr: float = LEARNING_RATE,
                use_attention: bool = USE_ATTENTION,
                print_every: int = PRINT_EVERY) -> None:
    """
    Minimal cross-entropy training on next-char prediction.

    To keep math transparent (and avoid a huge manual backprop through the whole
    attention stack), we *only* update W_fc (final linear -> logits).
    This is enough to see loss fall and probabilities shift.

    Educational tradeoff:
      • Pros: short, easy to follow, numerically stable.
      • Cons: model capacity is tiny (rest frozen), but OK for toy data.
    """
    W_fc = params["W_fc"]  # we will update in-place

    vocab_size = W_fc.shape[1]
    N = len(data_ids)
    if N <= BLOCK_SIZE + 1:
        print("⚠️ Not enough data to train (need > BLOCK_SIZE + 1 tokens).")
        return

    for epoch in range(epochs):
        # sample random slice of length BLOCK_SIZE+1
        i = random.randint(0, N - BLOCK_SIZE - 1)
        x = data_ids[i : i + BLOCK_SIZE]
        y = data_ids[i + 1 : i + BLOCK_SIZE + 1]  # target is shifted

        probs, h_last = forward_block(x, params, pos_enc, use_attention)
        target_idx = y[-1]
        # safety clamp
        if target_idx >= vocab_size:
            target_idx = vocab_size - 1

        # loss
        loss = cross_entropy_from_probs(probs, target_idx)

        # --- gradient wrt logits ---
        # For softmax+cross-entropy, dL/dlogits = probs; probs[target]-=1
        dlogits = probs.copy()
        dlogits[target_idx] -= 1.0  # (vocab_size,)

        # gradient wrt W_fc: logits = h_last @ W_fc
        # dL/dW_fc = outer(h_last, dlogits)
        grad_W_fc = np.outer(h_last, dlogits)  # (E, V)

        # update
        W_fc -= lr * grad_W_fc

        if epoch % print_every == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")


# ---------------------------------------------------------------------------
# 7. GENERATION (sample next chars)
# ---------------------------------------------------------------------------
def sample_next_char(probs: np.ndarray, topk: int = TOPK_GENERATE) -> int:
    """
    Pick a token ID from probability distribution.
    We top-k truncate to reduce pure noise in tiny models.
    """
    if topk is not None and topk > 0:
        top_idx = np.argpartition(probs, -topk)[-topk:]
        top_probs = probs[top_idx]
        top_probs = top_probs / top_probs.sum()
        return int(np.random.choice(top_idx, p=top_probs))
    else:
        # full multinomial
        return int(np.random.choice(len(probs), p=probs))


def generate_text(prompt: str,
                  params: Dict[str, np.ndarray],
                  stoi: Dict[str, int],
                  itos: Dict[int, str],
                  pos_enc: np.ndarray,
                  max_new_tokens: int = 128,
                  use_attention: bool = USE_ATTENTION) -> str:
    """
    Autoregressively generate characters after a prompt.
    """
    context = encode(prompt, stoi)
    # if prompt has unknown chars -> they are skipped; ensure context not empty
    if not context:
        context = [0]  # arbitrary fallback to first vocab char

    for _ in range(max_new_tokens):
        x = context[-BLOCK_SIZE:]
        probs, _ = forward_block(x, params, pos_enc, use_attention)
        nxt = sample_next_char(probs)
        context.append(nxt)

    return decode(context, itos)


# ---------------------------------------------------------------------------
# 8. STRUCTURED KNOWLEDGE PARSING
# ---------------------------------------------------------------------------
# We’ll parse simple bios like:
#   "John is a 28-year-old male. He is skilled in Python, Java."
#   "Aisha is a 25-year-old female born on 1999-01-02. She works as a Data Analyst."
#
# We'll capture: name, age, gender, dob (if present), skills (list), role/job

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
SKILL_PAT = re.compile(
    r"(skilled\s+in|expert\s+in|knows)\s+([^\.]+)", re.IGNORECASE
)

def parse_bio_line(line: str) -> Dict[str, Any]:
    """Extract structured fields from a single bio line."""
    info = {
        "name": None,
        "age": None,
        "gender": None,
        "dob": None,
        "skills": [],
        "role": None,
        "raw": line.strip(),
    }

    m = BIO_PAT.search(line)
    if m:
        info["name"]   = m.group("name").strip().split()[0]  # take first token
        info["age"]    = safe_int(m.group("age"))
        info["gender"] = norm_gender(m.group("gender"))
        info["dob"]    = m.group("dob")

    # role
    m2 = ROLE_PAT.search(line)
    if m2:
        info["role"] = m2.group(1).strip()

    # skills
    m3 = SKILL_PAT.search(line)
    if m3:
        skills_str = m3.group(2)
        skills = split_skills(skills_str)
        info["skills"] = skills

    # fallback name if start of line "Name ..."
    if not info["name"]:
        tokens = line.strip().split()
        if tokens:
            info["name"] = tokens[0]

    return info


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def norm_gender(g):
    if not g:
        return None
    g = g.lower()
    if g in ("m", "male", "man"):
        return "male"
    if g in ("f", "female", "woman"):
        return "female"
    return g


def split_skills(s: str) -> List[str]:
    """
    Split "Python, Java, and Kubernetes" -> ["Python", "Java", "Kubernetes"]
    """
    # replace 'and' with comma
    s = s.replace(" and ", ", ")
    parts = [p.strip(" .") for p in s.split(",")]
    parts = [p for p in parts if p]
    return parts


def build_people_knowledge(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse multi-line bios into a name->info dictionary.
    """
    people = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        info = parse_bio_line(line)
        if info["name"]:
            people[info["name"].lower()] = info
    return people


# ---------------------------------------------------------------------------
# 9. ANSWER QUERIES FROM STRUCTURED + GENERATIVE
# ---------------------------------------------------------------------------
def answer_about(name: str,
                 people_db: Dict[str, Dict[str, Any]],
                 params: Dict[str, np.ndarray],
                 stoi: Dict[str, int],
                 itos: Dict[int, str],
                 pos_enc: np.ndarray,
                 use_attention: bool = USE_ATTENTION,
                 use_model_flavor: bool = False) -> str:
    """
    Return a human-readable answer about `name`.
    If not found, optionally try generative fallback.
    """
    key = name.lower()
    if key in people_db:
        p = people_db[key]
        parts = []
        parts.append(f"{p['name']}")

        if p["age"] is not None:
            parts.append(f"is {p['age']} years old")

        if p["gender"]:
            parts.append(p["gender"])

        if p["dob"]:
            parts.append(f"(born {p['dob']})")

        if p["role"]:
            parts.append(f"and works as {p['role']}")

        if p["skills"]:
            parts.append(
                "with skills in " + ", ".join(p["skills"])
            )

        resp = " ".join(parts) + "."
        if use_model_flavor:
            # tiny model tries to continue; usually noisy but educational
            prompt = f" {name} "
            gen = generate_text(prompt, params, stoi, itos, pos_enc,
                                max_new_tokens=40, use_attention=use_attention)
            resp += "\n(model continuation) " + gen.strip()
        return resp

    # not found in structured DB → pure model try
    prompt = f"Tell me about {name}"
    gen = generate_text(prompt, params, stoi, itos, pos_enc,
                        max_new_tokens=80, use_attention=use_attention)
    return f"I don't have structured info on {name}. Here's what the tiny model says:\n{gen}"


# ---------------------------------------------------------------------------
# 10. ADD / UPDATE PERSON
# ---------------------------------------------------------------------------
def add_person(bio_line: str,
               people_db: Dict[str, Dict[str, Any]],
               full_text: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Append a new bio line to the full text corpus and update structured DB.
    Returns updated text + people_db.
    """
    full_text = full_text.rstrip() + "\n" + bio_line.strip() + "\n"
    info = parse_bio_line(bio_line)
    if info["name"]:
        people_db[info["name"].lower()] = info
    return full_text, people_db


# ---------------------------------------------------------------------------
# 11. SAVE / LOAD WEIGHTS
# ---------------------------------------------------------------------------
def save_weights(path: str, params: Dict[str, np.ndarray]) -> None:
    data = {k: v.tolist() for k, v in params.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_weights(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    return params


# ---------------------------------------------------------------------------
# 12. DEMO MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Base data you provided ---
    initial_data = """\
John is a 28-year-old male. He is skilled in Python, Java, and Kubernetes. He works as a DevOps Engineer.
Aisha is a 25-year-old female. She is an expert in Data Science, ML, and SQL. She works as a Data Analyst.
"""

    # Build structured people DB
    people_db = build_people_knowledge(initial_data)

    # Build vocab from current corpus
    stoi, itos = build_vocab(initial_data)
    vocab_size = len(stoi)

    # Encode to IDs
    data_ids = encode(initial_data, stoi)

    # Safety: ensure we have enough tokens
    if len(data_ids) < BLOCK_SIZE + 2:
        print("⚠️ Not enough data in initial_data to train meaningfully.")
        print("Add more bios!")
        exit(0)

    # Initialize weights
    params = init_weights(vocab_size, EMBED_SIZE)

    # Positional encodings
    pos_enc = positional_encoding(BLOCK_SIZE, EMBED_SIZE)

    # Train
    print("=== Training on initial data ===")
    train_model(data_ids, params, pos_enc,
                epochs=EPOCHS, lr=LEARNING_RATE,
                use_attention=USE_ATTENTION, print_every=PRINT_EVERY)

    # Query examples
    print("\n--- Query: John ---")
    print(answer_about("John", people_db, params, stoi, itos, pos_enc,
                       use_attention=USE_ATTENTION, use_model_flavor=False))

    print("\n--- Query: Aisha ---")
    print(answer_about("Aisha", people_db, params, stoi, itos, pos_enc,
                       use_attention=USE_ATTENTION, use_model_flavor=False))

    print("\n--- Query: Unknown person 'Ravi' (not yet added) ---")
    print(answer_about("Ravi", people_db, params, stoi, itos, pos_enc,
                       use_attention=USE_ATTENTION, use_model_flavor=True))

    # --- Add a new person ---
    new_bio = "Michael is a 30-year-old male. He works in Cloud Security and knows AWS and GCP."

    print("\n=== Adding new person: Michael ===")
    initial_data, people_db = add_person(new_bio, people_db, initial_data)

    # Rebuild vocab (new chars may have appeared: e.g., capital M, Cloud, GCP)
    stoi, itos = build_vocab(initial_data)
    vocab_size = len(stoi)

    # Re-encode
    data_ids = encode(initial_data, stoi)

    # Re-init + retrain (full rebuild approach; simple & clear)
    params = init_weights(vocab_size, EMBED_SIZE)
    pos_enc = positional_encoding(BLOCK_SIZE, EMBED_SIZE)

    print("=== Retraining on updated corpus (with Michael) ===")
    train_model(data_ids, params, pos_enc,
                epochs=EPOCHS, lr=LEARNING_RATE,
                use_attention=USE_ATTENTION, print_every=PRINT_EVERY)

    # Query Michael
    print("\n--- Query: Michael ---")
    print(answer_about("Michael", people_db, params, stoi, itos, pos_enc,
                       use_attention=USE_ATTENTION, use_model_flavor=False))

    # Save weights (optional)
    weights_path = "llm_toy_weights.json"
    save_weights(weights_path, params)
    print(f"\nWeights saved to {os.path.abspath(weights_path)}")
