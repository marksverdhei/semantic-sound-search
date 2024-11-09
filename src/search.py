
import faiss
import numpy as np

def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Using L2 distance
    index.add(embeddings)
    return index

def search_embedding(embedding, index, top_n=5):
    distances, indices = index.search(embedding, top_n)
    return distances[0], indices[0]