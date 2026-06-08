from sentence_transformers import SentenceTransformer
import chromadb
import os

# Configuration
BASE_URL = "localhost"
PORT = 8000
COLLECTION_NAME = "test_collection"
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Loading embedding model ({MODEL_NAME})...")
model = SentenceTransformer(MODEL_NAME)

def search_chroma(query):
    try:
        # 1. Initialize ChromaDB Client
        client = chromadb.HttpClient(host=BASE_URL, port=PORT)
        
        # 2. Get the collection
        collection = client.get_collection(name=COLLECTION_NAME)

        # 3. Convert query to vector
        query_vector = model.encode([query])[0].tolist()

        # 4. Query Chroma Database
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=1
        )

        if results['documents'] and len(results['documents'][0]) > 0:
            best_match_text = results['documents'][0][0]
            # Chroma with cosine space returns distance. Similarity is 1 - distance.
            distance = results['distances'][0][0]
            score = 1.0 - distance
            return best_match_text, score
        else:
            return "No relevant documents found.", 0.0

    except Exception as e:
        return f"Error connecting to ChromaDB: {e}", 0.0

def main():
    print("\n--- AI Semantic Search (ChromaDB Powered) ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Enter your search query: ")
        
        if query.lower() in ['exit', 'quit']:
            break
        
        if not query.strip():
            continue

        result, score = search_chroma(query)
        
        print("-" * 30)
        print(f"Result: {result}")
        print(f"Confidence Score: {score:.4f}")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    main()