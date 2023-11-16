import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

class SemanticProcessor:
    def __init__(self, embeddings_file_path):
        self.word_vectors = KeyedVectors.load_word2vec_format(embeddings_file_path)

    def calculate_similarity(self, phrase1, phrase2):
        # Implement your logic to calculate similarity (Cosine or Euclidean distance)
        pass

def batch_execution(phrases_file_path, embeddings_file_path):
    processor = SemanticProcessor(embeddings_file_path)
    df = pd.read_csv(phrases_file_path, encoding='ISO-8859-1')  # Specify the encoding

    similarities = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            similarity = processor.calculate_similarity(df['Phrase'][i], df['Phrase'][j])
            similarities.append({'Phrase1': df['Phrase'][i], 'Phrase2': df['Phrase'][j], 'Similarity': similarity})

    result_df = pd.DataFrame(similarities)
    return result_df

def on_the_fly_execution(input_phrase, phrases_file_path, embeddings_file_path):
    processor = SemanticProcessor(embeddings_file_path)
    df = pd.read_csv(phrases_file_path, encoding='ISO-8859-1')  # Specify the encoding

    df['Similarity'] = df['Phrase'].apply(lambda x: processor.calculate_similarity(input_phrase, x))

    closest_match = df.loc[df['Similarity'].idxmax()]
    return {'ClosestMatch': closest_match['Phrase'], 'Similarity': closest_match['Similarity']}
