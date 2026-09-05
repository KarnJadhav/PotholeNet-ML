# PotholeNet ML service — lightweight ONNX Runtime deployment image.
# Torch-free: uses ML_BACKEND=onnx (see app/main.py, app/inference_onnx.py).
# Built for Render, but the $PORT-binding pattern works on Cloud Run too.

FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless still needs a couple system libs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-onnx.txt .
RUN pip install --no-cache-dir -r requirements-onnx.txt

COPY app/ ./app/

ENV ML_BACKEND=onnx
# ML_HF_MODEL_REPO / ML_HF_MODEL_FILE / ML_CONFIDENCE_THRESHOLD / etc. are set
# as environment variables in the Render dashboard, not baked into the image —
# see .env.example for the full list.

# Render sets $PORT at runtime and expects the service to bind to it — do NOT
# hardcode 8000 here, the platform will route traffic to whatever $PORT says.
# Shell form (not exec form) is required so $PORT actually expands.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
