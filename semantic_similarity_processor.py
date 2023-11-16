import pandas as pd
import nltk
from gensim.models import KeyedVectors
from fuzzywuzzy import fuzz
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords

class SemanticProcessor:
    def __init__(self, embeddings_file_path):
        self.word_vectors = KeyedVectors.load_word2vec_format(embeddings_file_path)

    def clean_phrase(self, phrase):
        # Remove duplicates, outliers, and stopwords, and tokenize
        tokens = list(set(phrase.split()))
        tokens = [token for token in tokens if token.lower() not in stopwords]
        return tokens

    def find_closest_word(self, token):
        # Find the closest word using Levenshtein distance
        closest_word, _ = min(self.word_vectors.index2word, key=lambda w: fuzz.ratio(w, token))
        return closest_word

    def calculate_similarity(self, phrase1, phrase2):
        # Clean and tokenize phrases
        tokens1 = self.clean_phrase(phrase1)
        tokens2 = self.clean_phrase(phrase2)

        # Find word vectors for each token
        vectors1 = [self.word_vectors[token] if token in self.word_vectors else self.word_vectors[self.find_closest_word(token)] for token in tokens1]
        vectors2 = [self.word_vectors[token] if token in self.word_vectors else self.word_vectors[self.find_closest_word(token)] for token in tokens2]

        # Calculate cosine similarity between mean vectors of tokens
        if vectors1 and vectors2:
            vector1 = sum(vectors1) / len(vectors1)
            vector2 = sum(vectors2) / len(vectors2)
            similarity = 1 - cosine(vector1, vector2)
            return similarity
        else:
            return None

# Example of usage
if __name__ == "__main__":
    embeddings_path = r"C:\Users\DELL\Desktop\Galytix\vectors.csv"  # Update the path
    processor = SemanticProcessor(embeddings_path)

    # Example phrases
    phrase1 = "cleaning duplicates and stopwords"
    phrase2 = "remove duplicates and outliers"

    similarity = processor.calculate_similarity(phrase1, phrase2)
    print(f"Similarity between '{phrase1}' and '{phrase2}': {similarity}")
