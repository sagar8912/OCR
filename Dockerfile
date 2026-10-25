# Multi-stage / Production Dockerfile for OCR & Redaction Engine
FROM python:3.11-slim

# Labels for OCI container image metadata
LABEL org.opencontainers.image.title="OCR & Text Redaction Engine" \
      org.opencontainers.image.description="FastAPI service for CPU-optimized Optical Character Recognition and Inpainting-based text redaction" \
      org.opencontainers.image.authors="ML Engineering Team"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    OCR_MODEL_PATH="models/best_model.pt"

# Install system dependencies required by OpenCV and container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first for caching efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy runtime application files
COPY src/ ./src/
COPY models/ ./models/
COPY sample/ ./sample/

# Create non-root system user for security compliance
RUN useradd -m -u 10001 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose HTTP port
EXPOSE 8000

# Container healthcheck targeting FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command to start FastAPI with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
