from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = open("data/docs.txt").read().split("\n")
docs = [d for d in docs if d.strip()]

embeddings = np.load("embeddings.npy")

def search(query):

    query_embedding = model.encode([query])[0]

    scores = np.dot(embeddings, query_embedding)

    best_index = scores.argmax()

    return docs[best_index]


while True:

    q = input("\nAsk something: ")

    result = search(q)

    print("Answer:", result)