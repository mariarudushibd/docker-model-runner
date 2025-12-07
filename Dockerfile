FROM python:3.11-slim

WORKDIR /app

# Set environment variables for CPU optimization
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV TOKENIZERS_PARALLELISM=true
ENV TRANSFORMERS_OFFLINE=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch==2.4.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Copy and install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn[standard]==0.30.6 \
    transformers==4.45.0 pydantic==2.9.2 huggingface-hub==0.25.1 \
    optimum==1.23.0 onnxruntime==1.19.0

# Copy application code
COPY . .

# Create static directory
RUN mkdir -p /app/static
COPY static/ /app/static/

# Create non-root user for security
RUN useradd -m -u 1000 user
USER user

# Pre-download models during build for faster startup
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')" || true
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='distilgpt2')" || true

# Expose port
EXPOSE 7860

# Run with optimized settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
