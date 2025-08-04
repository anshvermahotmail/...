"""
==========================
Mini LLM Web Interface 🧠🌐
==========================

This single Python file provides:
- ✅ A backend API (FastAPI) for Q&A and learning
- ✅ A frontend UI (Flask) to ask questions and view full chat history
- ✅ Persistent memory using `facts.txt` (auto-created if not found)
- ✅ One-port serving on http://localhost:5000
- ✅ Runs both backend and frontend concurrently using threading

--------------------------
How it works:
--------------------------
1. User opens http://localhost:5000
2. Enters a question or a fact to learn (like "learn: Python is awesome")
3. Q&A is handled by a rule-based engine (no ML model required)
4. New facts are saved in `facts.txt` and used for future answers
5. Conversation history is maintained until page refresh

Dependencies:
- fastapi
- flask
- uvicorn
- requests

Install with:
pip install fastapi flask uvicorn requests

To run:
python llm_web_combined.py
"""

from flask import Flask, request, render_template_string
from fastapi import FastAPI, Request
from threading import Thread
import uvicorn
import requests
import os

FACT_FILE = "facts.txt"
conversation = []
facts = set()

# --- Load or create fact file ---
if not os.path.exists(FACT_FILE):
    with open(FACT_FILE, "w", encoding="utf-8") as f:
        f.write("The sky is blue.\nPython is a programming language.\n")
with open(FACT_FILE, "r", encoding="utf-8") as f:
    facts = set(line.strip() for line in f if line.strip())

# --- Backend: FastAPI for Q&A ---
app = FastAPI()

@app.post("/ask")
async def ask_question(req: Request):
    data = await req.json()
    question = data.get("question", "").strip()
    global facts

    if question.lower().startswith("learn:"):
        fact = question[6:].strip()
        if fact and fact not in facts:
            facts.add(fact)
            with open(FACT_FILE, "a", encoding="utf-8") as f:
                f.write(fact + "\n")
            return {"answer": f"✅ Learned: {fact}"}
        return {"answer": "⚠️ Already known or empty."}

    # Very simple LLM-like lookup from known facts
    response = next((f for f in facts if all(w.lower() in f.lower() for w in question.split())), None)
    answer = response or "🤖 Sorry, I don't know that yet."

    return {"answer": answer}

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

# --- Frontend: Flask UI ---
flask_app = Flask(__name__)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🧠 Mini LLM Chat</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f4f4f4; }
        textarea, input { width: 100%; font-size: 16px; }
        .msg { margin: 10px 0; padding: 10px; background: #fff; border-radius: 5px; }
    </style>
</head>
<body>
    <h2>💬 Mini LLM Chat</h2>
    <form method="post">
        <textarea name="question" rows="2" placeholder="Ask me anything... or say 'learn: Earth is round'"></textarea><br>
        <input type="submit" value="Send" />
    </form>
    <div>
        {% for user, bot in messages %}
        <div class="msg"><b>You:</b> {{ user }}</div>
        <div class="msg"><b>Bot:</b> {{ bot }}</div>
        {% endfor %}
    </div>
</body>
</html>
"""

@flask_app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        q = request.form.get("question", "").strip()
        if q:
            res = requests.post("http://127.0.0.1:8000/ask", json={"question": q})
            a = res.json().get("answer", "")
            conversation.append((q, a))
    return render_template_string(HTML_TEMPLATE, messages=conversation)

# --- Run both backend and frontend ---
if __name__ == "__main__":
    print("🚀 Launching Mini LLM Server at http://localhost:5000 ...")
    Thread(target=run_fastapi, daemon=True).start()
    flask_app.run(port=5000)
