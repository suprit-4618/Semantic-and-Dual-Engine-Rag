from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Semantic Search API")

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
BASE_URL = "localhost"
PORT = 8000
COLLECTION_NAME = "test_collection"

# Mount public directory for static files
os.makedirs("public", exist_ok=True)
app.mount("/static", StaticFiles(directory="public"), name="static")

print(f"Loading embedding model ({MODEL_NAME})...")
model = SentenceTransformer(MODEL_NAME)

class SearchQuery(BaseModel):
    query: str

@app.get("/")
def read_root():
    return FileResponse("public/index.html")

@app.post("/api/search")
def search(payload: SearchQuery):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        # Encode Query
        query_vector = model.encode([payload.query])[0].tolist()
        
        client = chromadb.HttpClient(host=BASE_URL, port=PORT)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        results_db = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        results = []
        context_texts = []
        if results_db['documents']:
            for i in range(len(results_db['documents'][0])):
                text = results_db['documents'][0][i]
                dist = results_db['distances'][0][i]
                score = 1.0 - dist
                results.append({
                    "text": text,
                    "score": float(score)
                })
                context_texts.append(text)
                
        # RAG Step: Generate answer using Groq (Strict Context)
        context_str = "\n\n".join(context_texts)
        rag_prompt = f"Answer the following question using ONLY the provided context. If the answer is not in the context, say 'I don't know based on the provided context.' Keep your answer extremely concise, strictly limited to 3 to 4 sentences.\n\nContext:\n{context_str}\n\nQuestion: {payload.query}\n\nAnswer:"
        
        # General Knowledge Step: Generate answer using Groq (Broad Knowledge)
        general_prompt = f"Answer the following question based on your general knowledge. Keep your answer extremely concise, strictly limited to 3 to 4 sentences.\n\nQuestion: {payload.query}\n\nAnswer:"
        
        try:
            import os
            GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables")
                
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # RAG Call
            groq_response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                headers=headers,
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": rag_prompt}],
                    "temperature": 0.0
                }, 
                timeout=30
            )
            groq_response.raise_for_status()
            llm_answer = groq_response.json()["choices"][0]["message"]["content"]
            
            # General Call
            general_response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                headers=headers,
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": general_prompt}],
                    "temperature": 0.7
                }, 
                timeout=30
            )
            general_response.raise_for_status()
            general_answer = general_response.json()["choices"][0]["message"]["content"]
            
        except Exception as api_error:
            print(f"Groq generation failed: {api_error}")
            llm_answer = "Could not generate an AI answer. There was an issue reaching the Groq API."
            general_answer = "Could not generate an AI answer."

        return {"results": results, "llm_answer": llm_answer, "general_answer": general_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/docs")
def get_docs():
    try:
        if not os.path.exists("data/docs.txt"):
            return {"docs": []}
        with open("data/docs.txt", "r") as f:
            docs = [line.strip() for line in f if line.strip()]
        return {"docs": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Note: Running on port 8500 to completely avoid old Endee cache/conflicts
    uvicorn.run("server:app", host="0.0.0.0", port=8500, reload=True)
