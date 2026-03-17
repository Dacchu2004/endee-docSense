from sentence_transformers import SentenceTransformer

# Load the model once when the file is imported
# all-MiniLM-L6-v2 produces 384-dimensional vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts: list) -> list:
    """
    Takes a list of strings, returns a list of vectors.
    Each vector is a list of 384 floats.
    normalize_embeddings=True makes cosine similarity work correctly.
    """
    return model.encode(texts, normalize_embeddings=True).tolist()