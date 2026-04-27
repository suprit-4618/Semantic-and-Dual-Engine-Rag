import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# --- Configuration ---
MODEL_NAME = "all-MiniLM-L6-v2"
DOCS_PATH = "data/docs.txt"
EMBEDDINGS_PATH = "embeddings.npy"

# --- Page Config ---
st.set_page_config(
    page_title="AI Semantic Search",
    page_icon="🔍",
    layout="centered"
)

# --- CSS for styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        color: #333333;
    }
    .result-card p {
        color: #333333 !important;
        margin-bottom: 10px;
    }
    .score-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.8em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Data & Model (Cached) ---
@st.cache_resource
def load_assets():
    # 1. Load Model
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Load Docs
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "r") as f:
            docs = [line.strip() for line in f if line.strip()]
    else:
        docs = []

    # 3. Load Embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        embeddings = None
        
    return model, docs, embeddings

# --- Search Logic (Pure NumPy for Portability) ---
def search_portable(query, model, docs, embeddings):
    if not docs or embeddings is None:
        return []
    
    # Encode Query
    query_vector = model.encode([query])[0]
    
    # Compute Cosine Similarity manually: (A dot B) / (||A|| * ||B||)
    # Since embeddings from SentenceTransformers are usually normalized, 
    # a simple dot product often works, but let's be precise:
    similarities = np.dot(embeddings, query_vector) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vector)
    )
    
    # Get top 3 indices
    top_indices = np.argsort(similarities)[::-1][:3]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": docs[idx],
            "score": float(similarities[idx])
        })
    return results

# --- UI Components ---
st.title("🔍 AI Semantic Search")
st.write("Find information based on **meaning**, not just keywords.")

model, docs, embeddings = load_assets()

# Help the user know what to search
with st.expander("📖 View Knowledge Base"):
    if docs:
        st.write(docs)
    else:
        st.warning("Knowledge base source not found.")

st.write("---")

query = st.text_input("Enter your search query:", placeholder="e.g., What is machine learning?")

if query:
    if not docs or embeddings is None:
        st.error("Assets missing! Please run `python embed.py` first.")
    else:
        with st.spinner("Searching..."):
            results = search_portable(query, model, docs, embeddings)
            
            if results:
                st.subheader("Top Results")
                for res in results:
                    st.markdown(f"""
                    <div class="result-card">
                        <p>{res['text']}</p>
                        <span class="score-badge">Confidence Score: {res['score']:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No relevant documents found.")

# --- Sidebar info ---
with st.sidebar:
    st.header("Quick Suggestions")
    st.markdown("- `What is machine learning?`\n- `Explain vector databases`\n- `What is NLP?`")
    st.write("---")
    st.header("Technical Stack")
    st.markdown("- **Model**: MiniLM-L6")
    st.markdown("- **Search**: Vector Similarity")
    st.markdown("- **Deploy**: Streamlit Cloud")
    st.info("This is a standalone AI search engine.")
