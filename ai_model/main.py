
print("Welcome to Personal AI Assistant")

print("My project is Federated Meta-Learning Framework")

user_name = input("Enter your name: ")

print("Hello", user_name)

print("How can I help you today?")
import re
import json
from pathlib import Path
from nlp_preprocessing import preprocess_text

# Load dataset
dataset_path = Path(__file__).parent / "intents.json"

with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("\nDataset Loaded Successfully!")

for intent in data["intents"]:
    print("\nIntent:", intent["tag"])

    for pattern in intent["patterns"]:
        print("Pattern:", pattern)

    print("Response:", intent["response"])
    import json
from pathlib import Path

# Load dataset
dataset_path = Path(__file__).parent / "intents.json"

with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("\nDataset Loaded Successfully!")

for intent in data["intents"]:
    print("\nIntent:", intent["tag"])

    for pattern in intent["patterns"]:
        print("Pattern:", pattern)

    print("Response:", intent["response"])


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Collect patterns
patterns = []
responses = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(preprocess_text(pattern))
        responses.append(intent["response"])

# Train text matching model
vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(patterns)

print("\nAI NLP Model Ready!")

while True:
    user_input = input("\nYou: ").lower().strip()
    user_input = re.sub(r'[^a-z0-9\s]', '', user_input)
    user_input = preprocess_text(user_input)

    if user_input in ["exit", "quit", "bye"]:
        print("AI: Goodbye!")
        break

    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(
        user_vector, pattern_vectors
    )

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score >= 0.25:
        print("AI:", responses[best_match])
    else:
        print("AI: Sorry, I don't understand.")




