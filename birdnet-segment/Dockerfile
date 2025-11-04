# syntax=docker/dockerfile:1.7
FROM python:3.11-bookworm

# Improve reproducibility and caching
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies (rarely change)
# RUN apt-get update \
#  && apt-get install -y --no-install-recommends ffmpeg \
#  && rm -rf /var/lib/apt/lists/*

# Copy only requirements for dependency caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy app source (changes more often)
COPY ./src .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]