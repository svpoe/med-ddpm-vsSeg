# Use Google's Deep Learning container as base
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13

# Set working directory
WORKDIR /app/vertex

# Copy requirements first for better caching
COPY requirements.txt . .

# Install build tools needed for scikit-learn and other C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    cython \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Set environment variables for proper imports
ENV PYTHONPATH=/app:/app/vertex
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Ensure all Python files are executable
RUN find /app -name "*.py" -exec chmod +x {} \;

# Create necessary directories
RUN mkdir -p /app/dataset /app/model /app/results /app/logs

# Default command (will be overridden by Vertex AI)
CMD ["python", "train_vs_seg_cloud.py"]
