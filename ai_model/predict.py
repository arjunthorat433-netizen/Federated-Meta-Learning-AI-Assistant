import joblib
from pathlib import Path

from nlp_preprocessing import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "ai_model.pkl"

model_data = joblib.load(MODEL_PATH)

vectorizer = model_data["vectorizer"]
pattern_vectors = model_data["pattern_vectors"]
intent_tags = model_data["intent_tags"]

print("AI Prediction System Ready!")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("AI: Goodbye!")
        break

    cleaned_input = preprocess_text(user_input)

    user_vector = vectorizer.transform([cleaned_input])

    similarity = cosine_similarity(
        user_vector, pattern_vectors
    )

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score >= 0.25:
        predicted_intent = intent_tags[best_match]
        print("AI Predicted Intent:", predicted_intent)
        print("Similarity Score:", round(score, 2))
    else:
        print("AI: Sorry, I don't understand.")