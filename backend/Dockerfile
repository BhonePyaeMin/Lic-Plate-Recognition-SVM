# NOT BUILT OR RUN during development (the Docker daemon was unavailable).
# Verify with: docker build -t plate-api . && docker run -p 8000:8000 plate-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# libgomp1: OpenMP runtime used by onnxruntime. libxcb1, libglib2.0-0, libgl1: required to import cv2.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libxcb1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
# Trained SVM weights (about 20 KB). Without this folder the API still runs; /health then reports svm: no_model.
COPY models ./models
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
# $PORT is set by most PaaS hosts; defaults to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
