# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 \
    torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Create directories for model caching
RUN mkdir -p /runpod-volume/models /app/outputs /tmp

# Copy the handler script
COPY rp_handler.py .

# Set model cache directory
ENV MODEL_CACHE_DIR=/runpod-volume/models \
    HF_HOME=/runpod-volume/models \
    TORCH_HOME=/runpod-volume/models \
    TRANSFORMERS_CACHE=/runpod-volume/models/transformers

# Optional: Pre-download models (comment out if you want faster builds)
# RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')" || true

# Test CUDA availability
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Start the handler
CMD ["python", "-u", "rp_handler.py"]