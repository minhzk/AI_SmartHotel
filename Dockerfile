FROM python:3.9-slim

# Install system dependencies including Git LFS
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Initialize Git LFS globally
RUN git lfs install --system

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies (BERT-only)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Setup Git LFS and pull model files if available
RUN if [ -d ".git" ]; then \
    git lfs install && \
    git lfs pull && \
    echo "✅ Git LFS files pulled"; \
    else \
    echo "ℹ️ No Git repository found, skipping LFS pull"; \
    fi || echo "⚠️ Git LFS pull failed or no LFS files"

# Create necessary directories
RUN mkdir -p models service logs

# Set environment variables for production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_DIR=/app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]