import json
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_preprocessing import preprocess_text

# Project Paths
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "intents.json"
MODEL_DIR = BASE_DIR / "models"

# Create Models Folder
MODEL_DIR.mkdir(exist_ok=True)

# Load Dataset
with open(DATASET_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare Training Data
patterns = []
intent_tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(preprocess_text(pattern))
        intent_tags.append(intent["tag"])

# Create TF-IDF Model
vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(patterns)

# Save Trained Model
model_data = {
    "vectorizer": vectorizer,
    "pattern_vectors": pattern_vectors,
    "patterns": patterns,
    "intent_tags": intent_tags
}

model_path = MODEL_DIR / "ai_model.pkl"

joblib.dump(model_data, model_path)

print("AI Model Training Completed!")
print("Total Training Patterns:", len(patterns))
print("Total Intents:", len(data["intents"]))
print("Model Saved Successfully!")
print("Saved Location:", model_path)