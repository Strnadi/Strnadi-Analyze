FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# the service needs only the pipeline, the segmenter and the shared constants (no torch, no training code)
COPY ./config.py ./runner.py ./
COPY ./serving ./serving
COPY ./processing/__init__.py ./processing/segment.py ./processing/

# export a trained run first:  python -m serving.export_onnx <run_dir> model.onnx
ARG HOST_MODEL_PATH=model.onnx
ENV MODEL_PATH=/app/model.onnx
COPY ${HOST_MODEL_PATH} ${MODEL_PATH}

ENTRYPOINT ["uvicorn", "runner:app", "--host", "0.0.0.0", "--port", "8000"]
