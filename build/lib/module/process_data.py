import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

class SemanticProcessor:
    def __init__(self, embeddings_file_path):
        self.word_vectors = KeyedVectors.load_word2vec_format(embeddings_file_path)

    def calculate_similarity(self, phrase1, phrase2):
        tokens1 = phrase1.split()
        tokens2 = phrase2.split()

        tokens1 = [token for token in tokens1 if token in self.word_vectors]
        tokens2 = [token for token in tokens2 if token in self.word_vectors]

        if not tokens1 or not tokens2:
            return None

        vector1 = self.word_vectors[tokens1].mean(axis=0)
        vector2 = self.word_vectors[tokens2].mean(axis=0)

        similarity = 1 - cosine(vector1, vector2)
        return similarity

def batch_execution(phrases_file_path, embeddings_file_path):
    processor = SemanticProcessor(embeddings_file_path)
    df = pd.read_csv(phrases_file_path)

    similarities = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            similarity = processor.calculate_similarity(df['Phrase'][i], df['Phrase'][j])
            similarities.append({'Phrase1': df['Phrase'][i], 'Phrase2': df['Phrase'][j], 'Similarity': similarity})

    result_df = pd.DataFrame(similarities)
    return result_df

def on_the_fly_execution(input_phrase, phrases_file_path, embeddings_file_path):
    processor = SemanticProcessor(embeddings_file_path)
    df = pd.read_csv(phrases_file_path)

    df['Similarity'] = df['Phrase'].apply(lambda x: processor.calculate_similarity(input_phrase, x))

    closest_match = df.loc[df['Similarity'].idxmax()]
    return {'ClosestMatch': closest_match['Phrase'], 'Similarity': closest_match['Similarity']}

if __name__ == "__main__":
    embeddings_path = "vectors.csv"  # Update the path
    phrases_file_path = r"C:\Users\DELL\Desktop\Galytix\phrases.csv"  # Update the path

    # On-the-fly Execution
    input_phrase = "user-input phrase"
    on_the_fly_result = on_the_fly_execution(input_phrase, phrases_file_path, embeddings_path)
    print("\nOn-the-fly Execution Result:")
    print(on_the_fly_result)

    # Batch Execution
    batch_result = batch_execution(phrases_file_path, embeddings_path)
    print("Batch Execution Result:")
    print(batch_result)
