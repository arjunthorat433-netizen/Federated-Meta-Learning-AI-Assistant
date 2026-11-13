import re
import nltk

from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):

    tokens = re.findall(r'\b\w+\b', text.lower())

    stop_words = {
        "i", "a", "an", "the", "is", "am",
        "are", "to", "of", "and"
    }

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    stemmer = PorterStemmer()

    stemmed_words = [
        stemmer.stem(word)
        for word in filtered_tokens
    ]

    lemmatizer = WordNetLemmatizer()

    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in filtered_tokens
    ]

    return " ".join(lemmatized_words)


if __name__ == "__main__":

    text = "I want a Personal AI Assistant!"

    print("Original Text:", text)

    result = preprocess_text(text)

    print("Preprocessed Text:", result)