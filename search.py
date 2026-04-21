from sentence_transformers import SentenceTransformer
from endee import Endee
import os

# Configuration
BASE_URL = "http://localhost:8080"
COLLECTION_NAME = "test_collection"
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Loading embedding model ({MODEL_NAME})...")
model = SentenceTransformer(MODEL_NAME)

def search_endee(query):
    try:
        # 1. Initialize Endee Client
        client = Endee()
        # The server requires the /api/v1 prefix
        client.set_base_url(f"{BASE_URL}/api/v1")
        
        # 2. Get the index
        idx = client.get_index(COLLECTION_NAME)

        # 3. Convert query to vector
        query_vector = model.encode([query])[0].tolist()

        # 4. Query Endee Vector Database
        results = idx.query(
            vector=query_vector,
            top_k=1
        )

        if results and len(results) > 0:
            best_match = results[0]
            # Endee SDK returns 'meta' and 'similarity' (or 'score')
            text = best_match.get("meta", {}).get("text", "No text metadata found")
            score = best_match.get("similarity", 0.0)
            return text, score
        else:
            return "No relevant documents found.", 0.0

    except Exception as e:
        return f"Error connecting to Endee: {e}", 0.0

def main():
    print("\n--- AI Semantic Search (Endee Powered) ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Enter your search query: ")
        
        if query.lower() in ['exit', 'quit']:
            break
        
        if not query.strip():
            continue

        result, score = search_endee(query)
        
        print("-" * 30)
        print(f"Result: {result}")
        print(f"Confidence Score: {score:.4f}")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    main()