# syntax=docker/dockerfile:1.7
FROM python:3.11-bookworm

# Keep Python & pip quiet and efficient
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set workdir early so COPY paths are stable and cacheable
WORKDIR /app

# OS packages change rarely → good cache layer
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Python deps: copy only requirements to maximize cache hits
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# App source: changes more often
COPY ./preprocess ./preprocess
COPY ./model ./model
COPY ./app.py .

# Model file: changes most often → copy last so earlier layers stay cached
ARG HOST_MODEL_PATH=model.keras
ENV MODEL_PATH=/app/model.keras
COPY ${HOST_MODEL_PATH} ${MODEL_PATH}

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
