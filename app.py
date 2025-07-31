import os;;;;
import sqlite3;;;;;
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from llama_cpp import Llama
from thefuzz import process, fuzz

# --- App Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default-fallback-key-for-dev")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# --- Load Models & Encoders ---
try:
    model = joblib.load("model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048)
except Exception as e:
    print(f"❌ Error loading a model file: {e}")
    exit()

# --- Keyword Lists ---
symptom_list = [
    'fever', 'cough', 'headache', 'sore throat', 'fatigue', 'vomiting', 'nausea',
    'diarrhea', 'body pain', 'loss of taste', 'chills', 'runny nose', 'congestion',
    'dizziness', 'sneezing', 'abdominal pain', 'muscle pain', 'loss of smell',
    'rash', 'joint pain', 'dehydration', 'sweating', 'weakness'
]
EMERGENCY_KEYWORDS = [
    'heart attack', 'chest pain', 'suicide', 'suicidal', 'cannot breathe',
    'bleeding out', 'stroke', 'seizure', 'unconscious', 'severe pain'
]

# --- Database Setup ---
DATABASE_NAME = "chatbot.db"
def init_db():
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
init_db()


@app.route("/")
def home():
    intro_message = (
        "Welcome! I am an AI assistant. I can help explore possibilities based on symptoms, "
        "but **I am not a doctor.** This is not a medical diagnosis. Please consult a healthcare professional for any health concerns."
    )
    return render_template("chat.html", intro=intro_message)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").lower().strip()
    session_id = session.sid

    if "symptoms" not in session:
        session["symptoms"] = []

    # --- NEW: Fuzzy Match for Emergency Keywords ---
    # We check for a high similarity score (e.g., 90) to be safe.
    best_emergency_match = process.extractOne(user_input, EMERGENCY_KEYWORDS, scorer=fuzz.token_set_ratio)
    if best_emergency_match and best_emergency_match[1] > 85:
        emergency_reply = (
            "This sounds like a potential medical emergency. **Please call your local emergency services immediately.** "
            "I am an AI and cannot provide assistance in this situation."
        )
        return jsonify({"reply": emergency_reply})

    # Store user message in DB
    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.execute("INSERT INTO chat_history (session_id, sender, message) VALUES (?, ?, ?)", 
                     (session_id, 'user', user_input))

    # --- Command Handling & Special Questions ---
    if any(word in user_input for word in ["reset", "start over", "clear"]):
        session["symptoms"] = []
        return jsonify({"reply": "🔄 Reset complete! You can start over with your symptoms."})

    if "who is vansh" in user_input or "who made you" in user_input:
        # ... (this part remains the same)
        prompt = "User asked who created you. Reply with a short, witty line explaining you were made by a student developer named Vansh."
        output = llm(prompt=prompt, max_tokens=50, stop=["\n"])
        reply = output["choices"][0]["text"].strip()
        return jsonify({"reply": reply})

    # --- NEW: Fuzzy Match for Symptoms ---
    # We use a slightly lower threshold here (e.g., 80) to be more flexible.
    extracted_symptoms = process.extractBests(user_input, symptom_list, scorer=fuzz.token_set_ratio, score_cutoff=80)
    matched_symptoms = [s[0] for s in extracted_symptoms] # Get just the symptom name

    # --- Main Logic Branch ---
    if not matched_symptoms:
        reply = handle_general_chat(user_input, session_id)
    else:
        reply = handle_symptom_logic(matched_symptoms)
    
    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.execute("INSERT INTO chat_history (session_id, sender, message) VALUES (?, ?, ?)", 
                     (session_id, 'bot', reply))

    return jsonify({"reply": reply})


def handle_general_chat(user_input, session_id):
    # ... (this function remains the same) ...
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT message FROM chat_history WHERE session_id=? AND sender='user' ORDER BY id DESC LIMIT 5", (session_id,))
        memory_lines = [row[0] for row in c.fetchall()][::-1]
        context = "\n".join(memory_lines)
    prompt = (
        "You are a funny and empathetic medical assistant chatbot. Your role is to have a "
        "brief, supportive conversation. **You must not give any medical advice, diagnosis, or suggestions for treatment, food, or drugs.** "
        "Keep your replies short.\n\n"
        f"Conversation History:\n{context}\n\nUser's latest message: {user_input}\n\nYour witty and safe reply:"
    )
    try:
        output = llm(prompt=prompt, max_tokens=60, stop=["\n", "User:"])
        reply = output["choices"][0]["text"].strip()
        if not reply or len(reply) < 3:
            reply = "I'm not quite sure how to respond to that, but I'm here to listen!"
    except Exception as e:
        print(f"LLM Error: {e}")
        reply = "Oops! My AI brain just had a little hiccup. Could you try asking that again?"
    return reply

def handle_symptom_logic(matched_symptoms):
    # ... (this function remains the same, but now receives fuzzy-matched symptoms) ...
    newly_added = []
    for s in matched_symptoms:
        if s not in session["symptoms"]:
            session["symptoms"].append(s)
            newly_added.append(s.replace("_", " "))
    current_symptoms = [s.replace("_", " ") for s in session["symptoms"]]
    if len(session["symptoms"]) < 2:
        return f"Got it! You've mentioned: {', '.join(newly_added)}. What other symptoms are you experiencing?"
    input_vector = [1 if s in session['symptoms'] else 0 for s in symptom_list]
    probs = model.predict_proba([input_vector])[0]
    top_index = np.argmax(probs)
    top_probability = probs[top_index]
    CONFIDENCE_THRESHOLD = 0.70 
    if top_probability >= CONFIDENCE_THRESHOLD:
        prediction = label_encoder.inverse_transform([top_index])[0]
        reply = (
            f"Based on your symptoms ({', '.join(current_symptoms)}), one possibility could be **{prediction}**. "
            "However, this is just a statistical suggestion and **not a medical diagnosis.** "
            "Please consult a real doctor for accurate advice. I will reset your symptoms now."
        )
    else:
        reply = (
            f"Thank you for sharing your symptoms ({', '.join(current_symptoms)}). "
            "Based on this combination, I am not confident enough to make a suggestion. "
            "It is always best to consult a doctor. I will reset your symptoms now."
        )
    session["symptoms"] = []
    return reply


if __name__ == "__main__":
    app.run(debug=True, port=5001)
