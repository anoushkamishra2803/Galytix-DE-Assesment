import gensim
from gensim.models import KeyedVectors

class WordEmbeddings:
    def __init__(self, binary_file_path, limit=1000000):
        self.word_vectors = KeyedVectors.load_word2vec_format(binary_file_path, binary=True, limit=limit)
        self.word_vectors.save_word2vec_format('vectors.csv')

if __name__ == "__main__":
    binary_file_location = r"C:\\Users\\DELL\\Desktop\\Galytix\\GoogleNews-vectors-negative300.bin" # Update the path
    word_embeddings = WordEmbeddings(binary_file_location)