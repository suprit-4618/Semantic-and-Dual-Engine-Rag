from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
with open("data/docs.txt", "r") as f:
    docs = f.read().split("\n")

# Remove empty lines
docs = [d for d in docs if d.strip() != ""]

# Generate embeddings
embeddings = model.encode(docs)

# Save embeddings
np.save("embeddings.npy", embeddings)

print("Embeddings generated successfully!")
