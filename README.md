# 🤖 AI-Powered Medical Chatbot (Local GenAI + Disease Diagnosis)

This is a fun, interactive, and **locally hosted AI chatbot** that:
- Diagnoses diseases from symptoms using a trained ML model 🧠
- Chats naturally using a local **Mistral** GenAI model (no API keys required!) 💬
- Remembers your name and symptoms using **SQLite** for persistent memory 🧷
- Has a humorous personality to keep things fun 😄

## 🏗 Tech Stack
- Python 3.x
- Flask (for web interface)
- SQLite3 (to store chat memory)
- Mistral 7B (via `llama-cpp-python` — for local GenAI)
- XGBoost (for disease prediction)
- HTML/CSS (chat interface)

## 🚀 Features
- Natural GenAI conversation (runs locally with GPU using RTX 3060)
- Trained on common medical symptoms (ML model using `symptoms.csv`)
- Remembers name, symptoms, and previous messages using a local database
- Automatically resets symptom session after diagnosis

## 🛠 Setup Instructions

```bash
# Clone this repo
git clone https://github.com/your-username/disease-chatbot-genai.git
cd disease-chatbot-genai

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install required packages
pip install -r requirements.txt

# Setup local database
python setup_db.py

# Run the app
python app.py
