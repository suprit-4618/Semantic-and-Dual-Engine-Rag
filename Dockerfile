FROM python:3.10-slim

# Create user with ID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Switch to non-root user
USER user

# Set home and paths
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user home
WORKDIR $HOME/app

# Copy requirements and install (chown to user)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (chown to user)
COPY --chown=user . .

# Generate Embeddings and build the embedded ChromaDB index during the Docker build
RUN python embed.py
RUN python insert_vectors.py

# Expose Hugging Face Space default port
EXPOSE 7860

# Run FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
