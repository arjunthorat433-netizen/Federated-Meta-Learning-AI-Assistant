
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_preprocessing import preprocess_text

# Load Dataset
file_path = Path(__file__).parent / "intents.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare Patterns and Intent Tags
patterns = []
intent_tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(preprocess_text(pattern))
        intent_tags.append(intent["tag"])

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(patterns)

# Test Dataset
test_data = [
    ("Hello there", "greeting"),
    ("Good morning", "greeting"),
    ("Hi AI", "greeting"),
    ("Remind me to study", "reminder"),
    ("Set a reminder", "reminder"),
    ("Remind me about work", "reminder"),
    ("Plan my day", "planning"),
    ("Make my daily schedule", "planning"),
    ("Create a study timetable", "planning"),
    ("xyz123", "unknown")
]

# Evaluate Model
correct = 0

print("\nAI MODEL EVALUATION")
print("------------------------")

for user_input, expected in test_data:

    cleaned_input = preprocess_text(user_input)

    user_vector = vectorizer.transform([cleaned_input])

    similarity = cosine_similarity(
        user_vector, pattern_vectors
    )

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score >= 0.25:
        predicted = intent_tags[best_match]
    else:
        predicted = "unknown"

    if predicted == expected:
        result = "PASS"
        correct += 1
    else:
        result = "FAIL"

    print("\nInput:", user_input)
    print("Expected:", expected)
    print("Predicted:", predicted)
    print("Result:", result)

# Calculate Accuracy
accuracy = (correct / len(test_data)) * 100

print("\n------------------------")
print("Correct Predictions:", correct)
print("Total Tests:", len(test_data))
print("Model Accuracy:", accuracy, "%")
