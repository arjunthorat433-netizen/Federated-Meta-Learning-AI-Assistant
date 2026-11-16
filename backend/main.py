import sys
import json
import random
from pathlib import Path

import requests
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from personalization.service import personalize
from ai_model.nlp_preprocessing import preprocess_text


app = FastAPI(
    title="Federated Meta-Learning AI Assistant API"
)


MODEL_PATH = ROOT / "ai_model" / "models" / "ai_model.pkl"
INTENTS_PATH = ROOT / "ai_model" / "intents.json"

model_data = joblib.load(MODEL_PATH)

vectorizer = model_data["vectorizer"]
pattern_vectors = model_data["pattern_vectors"]
intent_tags = model_data["intent_tags"]


with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents_data = json.load(f)


responses = {}

for intent in intents_data["intents"]:
    responses[intent["tag"]] = intent.get("responses", [])


general_answers = {
    "mumbai": "Mumbai is a major city in Maharashtra, India. It is located on the west coast of India and is the capital of Maharashtra.",
    "pune": "Pune is a major city in Maharashtra, India. It is known as an important educational and IT hub.",
    "india": "India is a country in South Asia. Its capital is New Delhi.",
    "maharashtra": "Maharashtra is a state in western India. Its capital is Mumbai and its winter capital is Nagpur.",
    "delhi": "New Delhi is the capital of India.",
    "python": "Python is a popular programming language used for web development, data science, artificial intelligence, automation, and many other applications.",
    "artificial intelligence": "Artificial Intelligence, or AI, is a field of computer science that enables computers to perform tasks that normally require human intelligence."
}


def get_general_answer(user_message):
    text = user_message.lower().strip()

    if "mumbai" in text:
        return general_answers["mumbai"]

    if "pune" in text:
        return general_answers["pune"]

    if "india" in text:
        return general_answers["india"]

    if "maharashtra" in text:
        return general_answers["maharashtra"]

    if "delhi" in text:
        return general_answers["delhi"]

    if "python" in text:
        return general_answers["python"]

    if "artificial intelligence" in text or text == "ai":
        return general_answers["artificial intelligence"]

    return None


def ask_ollama(user_message):
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:3b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Answer questions clearly and accurately."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "stream": False
            },
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]

        return "I could not generate an answer."

    except requests.exceptions.Timeout:
        return "The AI model is taking longer than expected. Please try again."

    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Please start Ollama and try again."

    except Exception as e:
        return f"AI error: {str(e)}"


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "AI Assistant Backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ai_model": "loaded",
        "personalization": "loaded",
        "ollama": "connected"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    user_message = request.message.strip()

    if not user_message:
        return {
            "user_message": user_message,
            "response": "Please enter a message."
        }

    processed_text = preprocess_text(user_message)

    query_vector = vectorizer.transform([processed_text])

    similarities = query_vector @ pattern_vectors.T

    best_index = similarities.argmax()

    best_score = similarities[0, best_index]

    intent = intent_tags[best_index]


    if best_score >= 0.25:

        intent_responses = responses.get(intent, [])

        if intent_responses:
            response = random.choice(intent_responses)
        else:
            response = ask_ollama(user_message)

    else:

        response = get_general_answer(user_message)

        if response is None:
            response = ask_ollama(user_message)


    if response is None:
        response = "Sorry, I could not generate an answer right now."


    return {
        "user_message": user_message,
        "intent": intent,
        "confidence": float(best_score),
        "response": response
    }


class FeedbackItem(BaseModel):
    features: list[float] = Field(min_length=8, max_length=8)
    liked: int = Field(ge=0, le=1)


class PersonalizeRequest(BaseModel):
    feedback: list[FeedbackItem]
    candidates: list[list[float]]


@app.post("/personalize")
def personalize_user(request: PersonalizeRequest):

    scores = personalize(
        [item.model_dump() for item in request.feedback],
        request.candidates
    )

    return {
        "personalization_scores": scores
    }