import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = re.findall(r"\b\w+\b", text.lower())

    stop_words = {
        "i", "a", "an", "the", "is",
        "am", "are", "to", "of", "and"
    }

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in filtered_tokens
    ]

    return " ".join(lemmatized_words)