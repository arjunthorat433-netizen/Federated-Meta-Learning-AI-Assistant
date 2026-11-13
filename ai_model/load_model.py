import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "ai_model.pkl"

# Load Trained Model
model_data = joblib.load(MODEL_PATH)

vectorizer = model_data["vectorizer"]
pattern_vectors = model_data["pattern_vectors"]
intent_tags = model_data["intent_tags"]

print("AI Model Loaded Successfully!")
print("Total Patterns:", len(intent_tags))
print("Model is Ready to Use!")