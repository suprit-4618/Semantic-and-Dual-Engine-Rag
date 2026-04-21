import numpy as np
import os
from endee import Endee, Precision

# Configuration
BASE_URL = "http://localhost:8080"
COLLECTION_NAME = "test_collection"

def insert_vectors():
    # Load documents
    if not os.path.exists("data/docs.txt"):
        print("Error: data/docs.txt not found.")
        return
    
    with open("data/docs.txt", "r") as f:
        docs = [line.strip() for line in f if line.strip()]

    # Load embeddings
    if not os.path.exists("embeddings.npy"):
        print("Error: embeddings.npy not found. Please run embed.py first.")
        return
    
    embeddings = np.load("embeddings.npy")

    if len(docs) != len(embeddings):
        print(f"Error: Number of docs ({len(docs)}) does not match number of embeddings ({len(embeddings)}).")
        return

    print(f"Connecting to Endee at {BASE_URL}...")
    
    try:
        # 1. Initialize Endee Client
        client = Endee()
        # The server requires the /api/v1 prefix
        client.set_base_url(f"{BASE_URL}/api/v1")
        
        # 2. Ensure index exists
        dimension = embeddings.shape[1]
        print(f"Ensuring index '{COLLECTION_NAME}' exists (dimension: {dimension})...")
        
        try:
            client.create_index(
                name=COLLECTION_NAME,
                dimension=dimension,
                space_type="cosine",
                precision=Precision.FLOAT32 # Using high precision for demo
            )
            print(f"Index '{COLLECTION_NAME}' created successfully.")
        except Exception as e:
            # If it already exists, we'll get an error, but we can continue
            print(f"Note: Index may already exist (status checked via creation attempt).")

        # 3. Get the index object
        idx = client.get_index(COLLECTION_NAME)

        # 4. Prepare payload for Endee SDK
        vectors_payload = []
        for i, emb in enumerate(embeddings):
            vectors_payload.append({
                "id": str(i),
                "vector": emb.tolist(),
                "meta": {"text": docs[i]}
            })

        print(f"Inserting {len(vectors_payload)} vectors...")
        
        # 5. Upsert vectors
        idx.upsert(vectors_payload)
        print("Successfully inserted vectors into Endee.")
            
    except Exception as e:
        print(f"Error during insertion: {e}")

if __name__ == "__main__":
    insert_vectors()