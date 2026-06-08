import numpy as np
import os
import chromadb

# Configuration
BASE_URL = "localhost"
PORT = 8000
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

    print(f"Connecting to ChromaDB at {BASE_URL}:{PORT}...")
    
    try:
        # 1. Connect to Embedded ChromaDB
        print("Connecting to local ChromaDB...")
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # 2. Ensure index exists
        dimension = embeddings.shape[1]
        print(f"Ensuring index '{COLLECTION_NAME}' exists (dimension: {dimension})...")
        
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Index '{COLLECTION_NAME}' ready.")

        # 3. Prepare payload for ChromaDB
        ids = [str(i) for i in range(len(embeddings))]
        metadatas = [{"text": doc} for doc in docs]
        embeddings_list = embeddings.tolist()

        print(f"Inserting {len(ids)} vectors...")
        
        # 4. Upsert vectors
        collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=docs
        )
        print("Successfully inserted vectors into ChromaDB.")
            
    except Exception as e:
        print(f"Error during insertion: {e}")

if __name__ == "__main__":
    insert_vectors()