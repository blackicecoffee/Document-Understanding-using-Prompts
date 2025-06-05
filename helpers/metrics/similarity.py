import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

print("Initializing Sentence Transformers...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def tf_idf(str1: str, str2: str):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str1, str2])
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity

def similarity_sentence_transformers(str1: str, str2: str):
    """Calculate similarity score using Sentence Transformers"""    
    sentences = [str1, str2]

    embeddings = model.encode(sentences)

    similarity = model.similarity(embeddings, embeddings).numpy()[0, 1]

    return similarity