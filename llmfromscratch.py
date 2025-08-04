import math
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ----------- Tokenization & Vocab Setup -----------

def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return [stoi[ch] for ch in text]

def decode(indices, itos):
    return ''.join([itos[i] for i in indices])

# ----------- Positional Embedding Setup -----------

def positional_encoding(seq_len, d_model):
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/d_model)))
    return pos_enc

# ----------- Attention Mechanism -----------

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(x, W_q, W_k, W_v, W_proj):
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    attn_scores = Q @ K.T / math.sqrt(Q.shape[-1])
    attn_weights = softmax(attn_scores)
    context = attn_weights @ V
    return context @ W_proj

# ----------- Model Forward Pass -----------

def forward(x, W_embed, W_q, W_k, W_v, W_proj):
    x_embed = W_embed[x]
    context = self_attention(x_embed, W_q, W_k, W_v, W_proj)
    logits = context.sum(axis=0)
    return softmax(logits)

# ----------- Loss Function -----------

def cross_entropy(probs, target_idx):
    return -math.log(probs[target_idx] + 1e-8)

# ----------- Training Loop -----------

def train(data, stoi, W_embed, W_q, W_k, W_v, W_proj, epochs=500):
    for epoch in range(epochs):
        idx = random.randint(0, len(data) - block_size - 1)
        x = data[idx:idx + block_size]
        y = data[idx + 1:idx + block_size + 1]

        probs = forward(x, W_embed, W_q, W_k, W_v, W_proj)
        loss = cross_entropy(probs, y[-1])

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")
    return W_embed, W_q, W_k, W_v, W_proj

# ----------- Utility to Add New Data -----------

def update_corpus(corpus, new_text):
    return corpus + "\n" + new_text.strip()

# ----------- MAIN CONFIG -----------

block_size = 8   # sequence length
embed_size = 16  # embedding vector length
num_heads = 1    # for future expansion

# Input base knowledge
initial_data = """
John is a 28-year-old male. He is skilled in Python, Java, and Kubernetes. He works as a DevOps Engineer.
Aisha is a 25-year-old female. She is an expert in Data Science, ML, and SQL. She works as a Data Analyst.
"""

# Build vocab
stoi, itos = build_vocab(initial_data)
vocab_size = len(stoi)

# Encode training data
corpus = encode(initial_data, stoi)

# Initialize weights
W_embed = np.random.randn(vocab_size, embed_size)
W_q = np.random.randn(embed_size, embed_size)
W_k = np.random.randn(embed_size, embed_size)
W_v = np.random.randn(embed_size, embed_size)
W_proj = np.random.randn(embed_size, vocab_size)

# Train the model
W_embed, W_q, W_k, W_v, W_proj = train(corpus, stoi, W_embed, W_q, W_k, W_v, W_proj)

# ----------- Ask About Someone -----------

def respond_to_query(name, corpus, stoi, itos):
    prompt = f"Tell me about {name}"
    tokens = encode(prompt[-block_size:], stoi)
    probs = forward(tokens, W_embed, W_q, W_k, W_v, W_proj)
    top_indices = np.argsort(probs)[-40:]  # return top likely characters
    print("Response:", decode(top_indices, itos))

respond_to_query("John", corpus, stoi, itos)

# ----------- Add More People -----------

new_bio = "Michael is a 30-year-old male. He works in Cloud Security and knows AWS and GCP."
initial_data = update_corpus(initial_data, new_bio)

# Rebuild vocab if new characters added
stoi, itos = build_vocab(initial_data)
vocab_size = len(stoi)
corpus = encode(initial_data, stoi)

# Reinitialize weights after vocab change
W_embed = np.random.randn(vocab_size, embed_size)
W_q = np.random.randn(embed_size, embed_size)
W_k = np.random.randn(embed_size, embed_size)
W_v = np.random.randn(embed_size, embed_size)
W_proj = np.random.randn(embed_size, vocab_size)

# Retrain
W_embed, W_q, W_k, W_v, W_proj = train(corpus, stoi, W_embed, W_q, W_k, W_v, W_proj)

# Test new data
respond_to_query("Michael", corpus, stoi, itos)
