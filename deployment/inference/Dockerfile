FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/inference/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Self-contained serverless image for RunPod.
COPY deployment/inference/serverless.py /app/serverless.py
COPY deployment/inference/main.py /app/main.py
COPY models /app/models
COPY preprocessing.py /app/preprocessing.py
COPY checkpoints/best_ocr_model.pth /app/checkpoints/best_ocr_model.pth
COPY checkpoints/best_restoration_model.pth /app/checkpoints/best_restoration_model.pth
COPY checkpoints/vocab.json /app/checkpoints/vocab.json

ENV PYTHONPATH="/app" \
    RESTORATION_CKPT="/app/checkpoints/best_restoration_model.pth" \
    OCR_CKPT="/app/checkpoints/best_ocr_model.pth" \
    VOCAB_PATH="/app/checkpoints/vocab.json" \
    DEVICE="cuda"

# RunPod Serverless entry point
CMD ["python", "-u", "/app/serverless.py"]
