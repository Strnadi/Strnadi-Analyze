# FROM tensorflow/tensorflow:latest-py3
FROM python:3.11-bookworm


ARG MODEL_PATH=model.keras
ENV MODEL_PATH='model.keras'


COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY ./preprocess ./preprocess
COPY ./model ./model
COPY ./app.py .

COPY ${MODEL_PATH} ./model.keras


EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000
