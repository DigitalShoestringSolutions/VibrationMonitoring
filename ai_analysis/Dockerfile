# Base stage for common dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install common Python dependencies
RUN pip install --no-cache-dir \
    celery==5.3.6 \
    redis==5.0.1 \
    python-dotenv==1.0.1 \
    influxdb-client==1.36.1

# API service stage
FROM base as api
# Install API-specific dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.2 \
    uvicorn==0.27.1 \
    pydantic==2.6.1 \
    python-multipart==0.0.9

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Default command for API service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Worker stage
FROM base as worker
# Install worker-specific dependencies
RUN pip install --no-cache-dir \
    tflite-runtime==2.14.0 \
    tensorflow==2.14.0 \
    numpy==1.24.3 \
    pandas==2.2.0 \
    scikit-learn==1.4.0 \
    flower==2.0.1 \
    matplotlib==3.10.0

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Default command for worker
CMD ["celery", "-A", "app.tasks", "worker", "--loglevel=info"] 