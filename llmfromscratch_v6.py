from flask import Flask, request, render_template_string
from fastapi import FastAPI, Request
from threading import Thread
import uvicorn
import requests
import os
import logging

FACT_FILE = "facts.txt"
REQUIREMENTS_FILE = "requirements.txt"
FLASK_PORT = 5000
API_PORT = 8000

conversation = []
facts = set()
debug_logs = []

# --- Create requirements.txt if not exists ---
if not os.path.exists(REQUIREMENTS_FILE):
    with open(REQUIREMENTS_FILE, "w") as req:
        req.write("flask\nfastapi\nuvicorn[standard]\nrequests\n")

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
    global facts, debug_logs

    debug_logs.clear()
    debug_logs.append(f"📥 Received question: '{question}'")

    if question.lower().startswith("learn:"):
        fact = question[6:].strip()
        debug_logs.append(f"🧠 Detected learning mode.")
        debug_logs.append(f"📘 Trying to learn: '{fact}'")

        if fact and fact not in facts:
            facts.add(fact)
            with open(FACT_FILE, "a", encoding="utf-8") as f:
                f.write(fact + "\n")
            debug_logs.append("✅ New fact learned and saved.")
            return {"answer": f"✅ Learned: {fact}", "log": debug_logs}
        debug_logs.append("⚠️ Fact already known or invalid.")
        return {"answer": "⚠️ Already known or empty.", "log": debug_logs}

    # Fact Lookup Logic
    debug_logs.append("🔍 Starting fact lookup...")
    words = question.lower().split()
    debug_logs.append(f"📊 Extracted words: {words}")
    debug_logs.append(f"📚 Scanning {len(facts)} facts for matches...")

    best_match = None
    highest_score = 0

    for fact in facts:
        fact_lower = fact.lower()
        matched_words = [w for w in words if w in fact_lower]
        score = len(matched_words)

        debug_logs.append(f"🔎 Checking: '{fact}'")
        debug_logs.append(f"   - Matched words: {matched_words}")
        debug_logs.append(f"   - Match score: {score}/{len(words)}")

        if score > highest_score:
            highest_score = score
            best_match = fact

    if best_match:
        debug_logs.append(f"✅ Best matching fact: '{best_match}' with score {highest_score}")
        return {"answer": best_match, "log": debug_logs}
    else:
        debug_logs.append("❌ No match found.")
        return {"answer": "🤖 Sorry, I don't know that yet.", "log": debug_logs}

# --- Run FastAPI backend ---
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="debug")

# --- Flask Frontend ---
flask_app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🧠 Mini LLM Chat</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f0f0f0; display: flex; }
        .container { width: 65%; }
        .chat-box {
            height: 60vh; overflow-y: auto;
            background: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;
        }
        .msg { margin: 10px 0; padding: 10px; background: #e6f7ff; border-radius: 5px; }
        .log-box {
            width: 30%; margin-left: 5%; background: #fff8dc; padding: 10px;
            border: 1px solid #ccc; border-radius: 5px; height: 80vh; overflow-y: auto;
        }
        textarea, input[type="submit"] {
            width: 100%; font-size: 16px;
        }
    </style>
    <script>
        window.onload = function() {
            const textarea = document.getElementById("question");
            textarea.focus();
            textarea.addEventListener("keydown", function(event) {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    this.form.submit();
                }
            });

            // Auto-scroll chat and log boxes
            const chatBox = document.querySelector(".chat-box");
            const logBox = document.querySelector(".log-box");
            chatBox.scrollTop = chatBox.scrollHeight;
            logBox.scrollTop = logBox.scrollHeight;
        };
    </script>
</head>
<body>
    <div class="container">
        <h2>💬 Mini LLM Chat</h2>
        <form method="post">
            <textarea id="question" name="question" rows="2" placeholder="Ask me anything... or say 'learn: Earth is round'"></textarea><br>
            <input type="submit" value="Send" />
        </form>
        <div class="chat-box">
            {% for user, bot in messages %}
                <div class="msg"><b>You:</b> {{ user }}</div>
                <div class="msg"><b>Bot:</b> {{ bot }}</div>
            {% endfor %}
        </div>
    </div>
    <div class="log-box">
        <h4>🔍 Debug Log</h4>
        <ul>
        {% for line in log %}
            <li>{{ line }}</li>
        {% endfor %}
        </ul>
    </div>
</body>
</html>
"""

@flask_app.route("/", methods=["GET", "POST"])
def index():
    debug_log = []
    if request.method == "POST":
        q = request.form.get("question", "").strip()
        if q:
            res = requests.post(f"http://127.0.0.1:{API_PORT}/ask", json={"question": q})
            data = res.json()
            a = data.get("answer", "")
            debug_log = data.get("log", [])
            conversation.append((q, a))
    return render_template_string(HTML_TEMPLATE, messages=conversation, log=debug_log)

# --- Run both backend and frontend ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(f"🚀 Launching Mini LLM at http://localhost:{FLASK_PORT}")
    Thread(target=run_fastapi, daemon=True).start()
    flask_app.run(debug=True, port=FLASK_PORT)
