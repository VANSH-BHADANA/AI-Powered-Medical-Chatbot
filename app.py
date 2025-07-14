from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import joblib
import numpy as np
import sqlite3
import os
from llama_cpp import Llama

# --- App Setup ---
app = Flask(__name__)
app.secret_key = "chatbot-secret-key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# --- Load ML Model & Encoder ---
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- Load Local Mistral/LLaMA Model ---
llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf")

# --- Symptom List ---
symptom_list = [
    'fever', 'cough', 'headache', 'sore_throat', 'fatigue', 'vomiting', 'nausea',
    'diarrhea', 'body_pain', 'loss_of_taste', 'chills', 'runny_nose', 'congestion',
    'dizziness', 'sneezing', 'abdominal_pain', 'muscle_pain', 'loss_of_smell',
    'rash', 'joint_pain', 'dehydration', 'sweating', 'weakness'
]

# --- Initialize SQLite ---
def init_db():
    conn = sqlite3.connect("chatbot.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            sender TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- Routes ---
@app.route("/")
def home():
    return render_template("chat.html", intro="👋 Hi! I'm your little AI Doctor 🤖. Tell me your symptoms or just say hi!")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").lower().strip()
    user_id = "default_user"

    # Store message in DB
    conn = sqlite3.connect("chatbot.db")
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (user_id, sender, message) VALUES (?, ?, ?)", (user_id, 'user', user_input))
    conn.commit()

    if "symptoms" not in session:
        session["symptoms"] = []

    # Reset command
    if any(word in user_input for word in ["reset", "start over", "clear"]):
        session["symptoms"] = []
        c.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return jsonify({"reply": "🔄 Reset done! Start again with your symptoms."})

    # Special Qs
    if "who is vansh" in user_input or "who made you" in user_input:
        conn.close()
        prompt = (
            "User asked who created the chatbot. Reply with a short, funny line explaining that it was made by a student named Vansh."
        )
        output = llm(prompt=prompt, max_tokens=50)
        reply = output["choices"][0]["text"].strip()
        return jsonify({"reply": reply})

    # Match symptoms
    matched = [s for s in symptom_list if s in user_input or s.replace("_", " ") in user_input]

    # If no match → fallback to GenAI
    if not matched:
        c.execute("SELECT message FROM chat_history WHERE user_id=? AND sender='user' ORDER BY id DESC LIMIT 5", (user_id,))
        memory_lines = [row[0] for row in c.fetchall()][::-1]
        context = "\n".join([f"{line}" for line in memory_lines])
        conn.close()

        prompt = (
            "You are a funny and helpful medical assistant. Respond to user inputs that are not symptoms "
            "with a witty, safe, and short reply. Avoid suggesting food, alcohol, or drugs.\n\n"
            f"Recent user inputs:\n{context}\nCurrent input: {user_input}\n\nReply:"
        )

        try:
            output = llm(prompt=prompt, max_tokens=60)
            reply = output["choices"][0]["text"].strip()
            if not reply or len(reply) < 3:
                reply = "I’m not sure what that means, but I’ll pretend I do 🤖😅"
        except Exception:
            reply = "Oops! My brain just glitched 🤯. Try again?"

        return jsonify({"reply": reply})

    # Add matched symptoms
    for s in matched:
        if s not in session["symptoms"]:
            session["symptoms"].append(s)

    current_symptoms = session["symptoms"]

    if len(current_symptoms) < 2:
        conn.close()
        return jsonify({
            "reply": f"Got it! You’ve mentioned: {', '.join(current_symptoms)}. Add more symptoms for a better guess!"
        })

    # Predict disease
    input_vector = [1 if s in current_symptoms else 0 for s in symptom_list]
    probs = model.predict_proba([input_vector])[0]
    top_index = np.argmax(probs)
    prediction = label_encoder.inverse_transform([top_index])[0]

    # Clear symptoms after prediction
    session["symptoms"] = []

    reply = f"🩺 Based on symptoms {', '.join(current_symptoms)}, you might have **{prediction}**. Take care and stay hydrated! 💧"

    # Save bot reply
    c.execute("INSERT INTO chat_history (user_id, sender, message) VALUES (?, ?, ?)", (user_id, 'bot', reply))
    conn.commit()
    conn.close()

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
