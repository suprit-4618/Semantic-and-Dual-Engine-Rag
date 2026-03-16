# AI Semantic Search using Endee Vector Database

## Project Overview

This project demonstrates an **AI-powered semantic search system** built using **vector embeddings** and the **Endee Vector Database**.

The system converts textual documents into high-dimensional vector embeddings using a transformer-based language model and retrieves the most relevant document based on semantic similarity.

Unlike traditional keyword search, semantic search understands the **meaning of the query**, enabling more accurate and context-aware results.

This project was developed as part of an assessment requiring the use of **Endee as the vector database backend**.

---

# Problem Statement

Traditional search systems rely on **keyword matching**, which fails when the query uses different wording than the stored documents.

Example:

Query:

```
What is artificial intelligence?
```

Document:

```
AI is the simulation of human intelligence by machines.
```

A keyword search might fail to match this query correctly.

Semantic search solves this by converting both the **query and documents into embeddings** and measuring **vector similarity**.

---

# Solution

This project implements a **semantic search pipeline** using the following steps:

1. Convert documents into vector embeddings
2. Store embeddings in a vector representation
3. Convert user query into embedding
4. Compute similarity between query and document vectors
5. Return the most relevant document

The project integrates **Endee vector database infrastructure** for vector indexing and storage.

---

# System Architecture

```
User Query
      ↓
SentenceTransformer Embedding Model
      ↓
Vector Representation (384-dimensional)
      ↓
Endee Vector Database
      ↓
Vector Similarity Search
      ↓
Most Relevant Document
```

---

# Workflow

### Step 1 — Document Preparation

Text documents are stored in:

```
data/docs.txt
```

Example:

```
Artificial Intelligence is the simulation of human intelligence by machines.

Machine learning is a subset of AI that allows computers to learn from data.

Deep learning uses neural networks with many layers.

Natural Language Processing enables computers to understand human language.

Vector databases store embeddings for semantic search.
```

---

### Step 2 — Embedding Generation

Documents are converted into embeddings using **SentenceTransformers**.

Script:

```
embed.py
```

Embedding model used:

```
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```
384
```

The generated embeddings are saved as:

```
embeddings.npy
```

---

### Step 3 — Semantic Search

The user enters a query.

Example:

```
What is AI?
```

The system:

1. Converts the query into a vector embedding
2. Computes similarity with stored document vectors
3. Returns the closest match.

Script used:

```
search.py
```

---

# Example Output

Example query:

```
Ask something: What is AI
```

Output:

```
Answer: Artificial Intelligence is the simulation of human intelligence by machines.
```

---

# Project Structure

```
ai-vector-project
│
├── data
│   └── docs.txt
│
├── embed.py
├── search.py
├── embeddings.npy
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python
* Sentence Transformers
* PyTorch
* NumPy
* Endee Vector Database

---

# Installation Guide

### 1. Clone the repository

```
git clone <your-repository-url>
cd ai-vector-project
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Generate embeddings

```
python embed.py
```

This will create:

```
embeddings.npy
```

---

### 4. Run semantic search

```
python search.py
```

Example usage:

```
Ask something: What is machine learning
```

---

# Key Concepts Demonstrated

This project demonstrates the following AI/ML concepts:

* Natural Language Processing
* Transformer-based embeddings
* Vector similarity search
* Semantic search systems
* Vector databases
* AI information retrieval

---

# Future Improvements

Potential improvements for this project include:

• Integrating real-time document ingestion
• Building a RAG (Retrieval Augmented Generation) chatbot
• Creating a web interface using FastAPI or Streamlit
• Scaling vector indexing using distributed Endee deployment
• Adding support for PDF or web document ingestion

---

# Learning Outcomes

Through this project the following concepts were explored:

* Embedding generation using transformer models
* Vector similarity search
* Semantic information retrieval
* Integration with vector database infrastructure

---

# Author

**Suprit Lenkennavar**

AI & Data Science Student
Interested in AI systems, intelligent software, and data-driven applications.
