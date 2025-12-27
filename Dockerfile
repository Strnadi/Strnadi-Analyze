FROM python:3.12-slim

WORKDIR /app
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./runner.py .

ARG HOST_MODEL_PATH=model.tflite
ENV MODEL_PATH=/app/model.tflite
COPY ${HOST_MODEL_PATH} ${MODEL_PATH}

ENTRYPOINT ["uvicorn", "runner:app", "--host", "0.0.0.0", "--port", "8000"]
