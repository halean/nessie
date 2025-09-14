FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (build tools, OpenMP, Python, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    cmake \
    libomp-dev \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install CUDA-enabled PyTorch first (compatible with Flair 0.11.x)
RUN python3 -m pip install \
      --index-url https://download.pytorch.org/whl/cu113 \
      torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113

## Install remaining Python dependencies from static requirements files
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install -r /app/requirements.txt

COPY requirements-dev.txt /app/requirements-dev.txt
RUN python3 -m pip install -r /app/requirements-dev.txt

RUN python3 -m pip install huggingface_hub==0.10.0 cleanlab==2.1.0
# Copy project source after deps for better layer caching
COPY . /app

# Install the package itself without re-resolving deps
RUN python3 -m pip install --no-deps .



# Create non-root user
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

CMD ["python3"]
