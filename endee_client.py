import requests
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = "http://localhost:8080"

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = open("data/docs.txt").read().split("\n")
docs = [d for d in docs if d.strip()]

embeddings = np.load("embeddings.npy")

print("Uploading vectors...")

payload = []

for i, emb in enumerate(embeddings):
    payload.append({
        "id": i,
        "values": emb.tolist(),
        "metadata": {"text": docs[i]}
    })

r = requests.post(
    f"{BASE}/collections/test_collection/vectors",
    json={"vectors": payload}
)

print("Upload response:", r.text)

while True:

    q = input("\nAsk a question: ")

    vec = model.encode([q])[0].tolist()

    r = requests.post(
        f"{BASE}/collections/test_collection/search",
        json={
            "vector": vec,
            "top_k": 1
        }
    )

    print("\nResult:")
    print(r.text)