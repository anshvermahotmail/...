import os
import re
import math
import random
import json
import socket
import time
import threading
from typing import Dict, List, Tuple, Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import webbrowser


#######################
#  Config & RNG seeds #
#######################
random.seed(42)
np.random.seed(42)

BLOCK_SIZE = 16
EMBED_SIZE = 32
USE_ATTENTION = True
LEARNING_RATE = 1e-2
EPOCHS = 1000
PRINT_EVERY = 100
TOPK_GENERATE = 5

STATIC_DIR = "static"
INDEX_FILEPATH = os.path.join(STATIC_DIR, "index.html")

#####################
# Create requirements.txt if missing
if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write("fastapi\nuvicorn\nnumpy\n")

#########################
# Minimal Frontend HTML
FRONTEND_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Mini LLM Fact Chat</title>
  <style>
    html,body {height:100%;margin:0;padding:0;font-family:sans-serif;background:#f5f5fa;}
    #container {max-width:960px;margin:0 auto;display:flex;flex-direction:row;}
    #chatBox {
      width:60vw; min-width:320px; max-width:700px; margin-top:30px;
      height:72vh; border:1px solid #ccc; background:white; border-radius:7px;
      padding:24px 16px 70px 16px; box-sizing:border-box;
      overflow-y:auto; font-size:18px; position:relative;
    }
    #formArea {
      position:absolute; left:10px; bottom:18px; width:90%;
    }
    #promptBox {
      font-size:18px; width:74%; padding:7px;
      border-radius:3px; border:1px solid #aaa;
    }
    #sendBtn {
      font-size:18px; margin-left:9px; padding:7px 18px;
      background:#1b56ae; color:#fff; border:none; border-radius:3px;
    }
    #debugPanel {
      width:34vw; min-width:240px; max-width:420px; padding:12px 10px;
      background:#23272e; color:#d1d9e7; font-size:14px; margin-left:34px;
      height:88vh; margin-top:10px; overflow-y:auto; border-radius:10px;
    }
    .chatmsg { margin:0 0 20px 0; }
    .user { color:#0b3990; }
    .bot { color:#191919; }
    h3 { margin-top:0; }
  </style>
</head>
<body>
  <div id="container">
      <div id="chatBox">
        <h3>🔎 Ask: “John” or “Aisha”…</h3>
        <div id="msgs"></div>
        <form id="formArea" onsubmit="sendMsg();return false;">
          <input autofocus id="promptBox" autocomplete="off" type="text" placeholder="Type a name…" />
          <button id="sendBtn" type="submit">Send</button>
        </form>
      </div>
      <div id="debugPanel"><b>Tech Log</b> <div id="dlog"></div></div>
  </div>
<script>
  function appendMsg(user, msg) {
    let msgs = document.getElementById("msgs");
    let div = document.createElement("div");
    div.className = "chatmsg";
    div.innerHTML = "<b class='" + user + "'>" + (user==="user"?"You: ":"Bot: ") + "</b>" + msg;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }
  function showLog(log) {
    let dlog = document.getElementById("dlog");
    dlog.innerHTML = "";
    log.forEach(line => {
      let p = document.createElement("div");
      p.textContent = line;
      dlog.appendChild(p);
    });
    dlog.scrollTop = dlog.scrollHeight;
  }
  document.getElementById("promptBox").addEventListener("keydown", function(e) {
    if(e.key==="Enter") {
      sendMsg();
      e.preventDefault();
    }
  });
  function refocus() { document.getElementById('promptBox').focus(); }
  window.onload = refocus;
  async function sendMsg() {
    let box = document.getElementById("promptBox");
    let usertxt = box.value.trim();
    if(!usertxt) return;
    appendMsg("user", usertxt);
    box.value = '';
    fetch("/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({query: usertxt})
    })
    .then(r=>r.json())
    .then(js => {
      appendMsg("bot", js.answer.replace(/\\n/g,"<br>"));
      showLog(js.log);
      refocus();
    });
  }
</script>
</body>
</html>
"""

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
if not os.path.exists(INDEX_FILEPATH):
    with open(INDEX_FILEPATH, "w", encoding="utf-8") as f:
        f.write(FRONTEND_HTML)

###################
# Minimal LLM + facts as in your original plus improved answer_about

def build_vocab(text: str) -> Tuple[Dict[str,int], Dict[int,str]]:
    chars = sorted(set(text))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return stoi, itos

def encode(text: str, stoi: Dict[str,int]) -> List[int]:
    return [stoi[ch] for ch in text if ch in stoi]

def decode(indices: List[int], itos: Dict[int,str]) -> str:
    return ''.join(itos[i] for i in indices if i in itos)

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pos_enc = np.zeros((seq_len,d_model),dtype=np.float32)
    for pos in range(seq_len):
        for i in range(0,d_model,2):
            angle=pos/(10000**((2*i)/d_model))
            pos_enc[pos,i]=math.sin(angle)
            if i+1<d_model:
                pos_enc[pos,i+1]=math.cos(angle)
    return pos_enc

def softmax(x: np.ndarray, axis: int=-1) -> np.ndarray:
    x=x-np.max(x,axis=axis,keepdims=True)
    e_x=np.exp(x)
    return e_x/e_x.sum(axis=axis,keepdims=True)

def cross_entropy_from_probs(probs: np.ndarray,target_idx:int) -> float:
    target_idx=int(target_idx)
    if target_idx<0 or target_idx>=probs.size:
        target_idx=min(max(target_idx,0),probs.size-1)
    return -math.log(float(probs[target_idx])+1e-8)

def init_weights(vocab_size:int, embed_size:int=EMBED_SIZE) -> Dict[str,np.ndarray]:
    scale=0.02
    return {
        "W_embed":np.random.randn(vocab_size,embed_size)*scale,
        "W_q":np.random.randn(embed_size,embed_size)*scale,
        "W_k":np.random.randn(embed_size,embed_size)*scale,
        "W_v":np.random.randn(embed_size,embed_size)*scale,
        "W_o":np.random.randn(embed_size,embed_size)*scale,
        "W_fc":np.random.randn(embed_size,vocab_size)*scale,
    }

def forward_block(x_ids: List[int], params: Dict[str,np.ndarray], pos_enc: np.ndarray, use_attention=True) -> Tuple[np.ndarray,np.ndarray]:
    W_embed, W_q, W_k, W_v, W_o, W_fc = params["W_embed"], params["W_q"], params["W_k"], params["W_v"], params["W_o"], params["W_fc"]
    x_embed = W_embed[x_ids]
    T = x_embed.shape[0]
    x_embed = x_embed + pos_enc[:T,:]
    if use_attention:
        Q = x_embed @ W_q
        K = x_embed @ W_k
        V = x_embed @ W_v
        scores = (Q @ K.T) / math.sqrt(Q.shape[-1])
        mask = np.triu(np.ones((T,T),dtype=bool),k=1)
        scores = np.where(mask,-1e10,scores)
        attn = softmax(scores,axis=-1)
        context = attn @ V
        h = context @ W_o
    else:
        h_mean = x_embed.mean(axis=0,keepdims=True)
        h = np.repeat(h_mean,T,axis=0)
    h_last = h[-1]
    logits = h_last @ W_fc
    probs = softmax(logits,axis=-1)
    return probs,h_last

def train_model(data_ids: List[int], params: Dict[str,np.ndarray], pos_enc: np.ndarray,
                epochs=EPOCHS, lr=LEARNING_RATE, use_attention=True, print_every=PRINT_EVERY) -> None:
    W_fc = params["W_fc"]
    vocab_size = W_fc.shape[1]
    N = len(data_ids)
    if N <= BLOCK_SIZE + 1:
        print("⚠️ Not enough data to train (need > BLOCK_SIZE + 1 tokens).")
        return
    for epoch in range(epochs):
        i = random.randint(0, N - BLOCK_SIZE - 1)
        x = data_ids[i:i+BLOCK_SIZE]
        y = data_ids[i+1:i+BLOCK_SIZE+1]
        probs, h_last = forward_block(x, params, pos_enc, use_attention)
        target_idx = y[-1]
        if target_idx >= vocab_size: target_idx = vocab_size-1
        loss = cross_entropy_from_probs(probs,target_idx)
        dlogits = probs.copy()
        dlogits[target_idx] -= 1.0
        grad_W_fc = np.outer(h_last,dlogits)
        W_fc -= lr * grad_W_fc
        if epoch % print_every == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

def sample_next_char(probs: np.ndarray, topk: int=TOPK_GENERATE) -> int:
    if topk and topk > 0:
        top_idx = np.argpartition(probs,-topk)[-topk:]
        top_probs=probs[top_idx]
        top_probs = top_probs/top_probs.sum()
        return int(np.random.choice(top_idx,p=top_probs))
    else:
        return int(np.random.choice(len(probs),p=probs))

def generate_text(prompt: str, params: Dict[str,np.ndarray], stoi: Dict[str,int], itos: Dict[int,str],
                  pos_enc: np.ndarray, max_new_tokens=128, use_attention=True) -> str:
    context = encode(prompt, stoi)
    if not context: context = [0]
    for _ in range(max_new_tokens):
        x = context[-BLOCK_SIZE:]
        probs, _ = forward_block(x, params, pos_enc, use_attention)
        nxt = sample_next_char(probs)
        context.append(nxt)
    return decode(context, itos)

BIO_PAT = re.compile(
    r"""^\s*(?P<name>[A-Z][A-Za-z0-9_ ]+?)\s+is\s+(?:a\s+)?(?P<age>\d{1,3})[- ]?year[- ]?old\s+
    (?P<gender>male|female|m|f|man|woman)?.*?(?:born\s+on\s+(?P<dob>\d{4}-\d{2}-\d{2}))?.*$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL)

ROLE_PAT = re.compile(r"works\s+as\s+a?n?\s+([^\.]+)", re.IGNORECASE)
SKILL_PAT = re.compile(r"(skilled\s+in|expert\s+in|knows)\s+([^\.]+)", re.IGNORECASE)

def parse_bio_line(line: str) -> Dict[str, Any]:
    info = {"name": None, "age": None, "gender": None, "dob": None, "skills": [], "role": None, "raw": line.strip()}
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
        info["skills"] = split_skills(m3.group(2))
    if not info["name"]:
        tokens = line.strip().split()
        if tokens: info["name"] = tokens[0]
    return info

def safe_int(x):
    try: return int(x)
    except: return None

def norm_gender(g):
    if not g: return None
    g = g.lower()
    if g in ("m","male","man"): return "male"
    if g in ("f","female","woman"): return "female"
    return g

def split_skills(s):
    s = s.replace(" and ", ", ")
    parts = [p.strip(" .") for p in s.split(",")]
    return [p for p in parts if p]

def build_people_knowledge(text):
    people = {}
    for line in text.strip().splitlines():
        line=line.strip()
        if not line: continue
        info = parse_bio_line(line)
        if info["name"]: people[info["name"].lower()] = info
    return people

def answer_about(name, people_db, params, stoi, itos, pos_enc, use_attention=USE_ATTENTION, use_model_flavor=True) -> Tuple[str,List[str]]:
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
            gen = generate_text(prompt, params, stoi, itos, pos_enc, max_new_tokens=40, use_attention=use_attention)
            logs.append(f"Model generated continuation: {gen.strip()}")
            full_resp = fact_resp + "\n(model continuation) " + gen.strip()
        else:
            full_resp = fact_resp
        logs.append("Final combined response assembled.")
        return full_resp, logs
    else:
        logs.append(f"No structured info found for '{name}'. Using pure generative fallback.")
        prompt = f"Tell me about {name}"
        gen = generate_text(prompt, params, stoi, itos, pos_enc, max_new_tokens=80, use_attention=use_attention)
        logs.append(f"Model generation output: {gen.strip()}")
        fallback_resp = f"I don't have structured info on {name}. Here's what the tiny model says:\n{gen}"
        logs.append("Final fallback response assembled.")
        return fallback_resp, logs

def add_person(bio_line: str, people_db: Dict[str, Dict[str,Any]], full_text: str) -> Tuple[str, Dict[str,Dict[str,Any]]]:
    full_text = full_text.rstrip() + "\n" + bio_line.strip() + "\n"
    info = parse_bio_line(bio_line)
    if info["name"]:
        people_db[info["name"].lower()] = info
    return full_text, people_db

def save_weights(path: str, params: Dict[str,np.ndarray]) -> None:
    data = {k: v.tolist() for k,v in params.items()}
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f)

def load_weights(path: str) -> Dict[str,np.ndarray]:
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    return {k: np.array(v,dtype=np.float32) for k,v in data.items()}


##############
# FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index(): return FileResponse(INDEX_FILEPATH)

# Global vars for demo
initial_data = """John is a 28-year-old male. He is skilled in Python, Java, and Kubernetes. He works as a DevOps Engineer.
Aisha is a 25-year-old female. She is an expert in Data Science, ML, and SQL. She works as a Data Analyst.
"""
people_db = build_people_knowledge(initial_data)
stoi, itos = build_vocab(initial_data)
vocab_size = len(stoi)
data_ids = encode(initial_data, stoi)
params = init_weights(vocab_size, EMBED_SIZE)
pos_enc = positional_encoding(BLOCK_SIZE, EMBED_SIZE)
train_model(data_ids, params, pos_enc, epochs=EPOCHS, lr=LEARNING_RATE, use_attention=USE_ATTENTION, print_every=PRINT_EVERY)

@app.post("/ask")
async def ask_api(req: Request):
    data = await req.json()
    query = data.get("query", "").strip()
    if not query:
        return JSONResponse({"answer": "Please ask a question.", "log": ["No query received."]})
    answer, logs = answer_about(query, people_db, params, stoi, itos, pos_enc, use_attention=USE_ATTENTION, use_model_flavor=True)
    return JSONResponse({"answer": answer, "log": logs})

##################################
# Utility: find free port
def find_free_port(start=8000):
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                port += 1

PORT = find_free_port(8000)
URL = f"http://localhost:{PORT}"

def open_browser_when_ready(url):
    import urllib.request
    for _ in range(80):
        try:
            urllib.request.urlopen(url)
            time.sleep(0.1)
            break
        except:
            time.sleep(0.15)
    webbrowser.open(url)

if __name__ == "__main__":
    print(f"Starting server on {URL}")
    threading.Thread(target=open_browser_when_ready, args=(URL,), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, reload=False)
