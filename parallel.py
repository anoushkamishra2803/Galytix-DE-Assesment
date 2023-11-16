import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from gensim.models import KeyedVectors
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine

# Assuming you have already downloaded Word2Vec embeddings and created the SemanticProcessor class

class SemanticProcessor:
    def __init__(self, embeddings_file_path):
        self.word_vectors = KeyedVectors.load_word2vec_format(embeddings_file_path)

    def clean_phrase(self, phrase):
        tokens = list(set(phrase.split()))
        tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]
        return tokens

    def find_closest_word(self, token):
        closest_word, _ = min(self.word_vectors.index2word, key=lambda w: fuzz.ratio(w, token))
        return closest_word

    def calculate_similarity(self, phrase1, phrase2):
        tokens1 = self.clean_phrase(phrase1)
        tokens2 = self.clean_phrase(phrase2)

        vectors1 = [self.word_vectors[token] if token in self.word_vectors else self.word_vectors[self.find_closest_word(token)] for token in tokens1]
        vectors2 = [self.word_vectors[token] if token in self.word_vectors else self.word_vectors[self.find_closest_word(token)] for token in tokens2]

        if vectors1 and vectors2:
            vector1 = sum(vectors1) / len(vectors1)
            vector2 = sum(vectors2) / len(vectors2)
            similarity = 1 - cosine(vector1, vector2)
            return similarity
        else:
            return None

def process_chunk(chunk, embeddings_path):
    processor = SemanticProcessor(embeddings_path)
    
    similarities = []
    for i in range(len(chunk)):
        for j in range(i+1, len(chunk)):
            similarity = processor.calculate_similarity(chunk['Phrase'][i], chunk['Phrase'][j])
            similarities.append({'Phrase1': chunk['Phrase'][i], 'Phrase2': chunk['Phrase'][j], 'Similarity': similarity})

    return similarities

def process_data_in_parallel(phrases_file_path, embeddings_path, chunk_size=1000, max_workers=4):
    df_chunks = pd.read_csv(phrases_file_path, chunksize=chunk_size, encoding='ISO-8859-1')


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process each chunk in parallel
        process_chunk_partial = partial(process_chunk, embeddings_path=embeddings_path)
        results = executor.map(process_chunk_partial, df_chunks)

    # Combine the results from all chunks
    all_results = []
    for result in results:
        all_results.extend(result)

    return pd.DataFrame(all_results)

# Example of usage
if __name__ == "__main__":
    embeddings_path = "vectors.csv"  # Update the path
    phrases_file_path = "phrases.csv"  # Update the path

    result_df = process_data_in_parallel(phrases_file_path, embeddings_path)
    print(result_df)
