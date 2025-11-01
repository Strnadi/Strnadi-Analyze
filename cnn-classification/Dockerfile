FROM python:3.11-bookworm

ARG MODEL_PATH=model.keras
ENV MODEL_PATH=$MODEL_PATH

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY ./preprocess ./preprocess
COPY ./model ./model
COPY ./app.py .

COPY ${MODEL_PATH} .


EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
