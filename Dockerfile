FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Generate Embeddings and build the embedded ChromaDB index during the Docker build
RUN python embed.py
RUN python insert_vectors.py

# Expose Hugging Face Space default port
EXPOSE 7860

# Run FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
