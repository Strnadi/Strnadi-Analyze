# Build from the repository root:  docker build -f vastai-serverless/Dockerfile --build-arg HOST_MODEL_PATH=model.onnx .
#
# The previous version of this file copied `model_server.py` and `start.sh`, neither of which existed in the
# repository, and installed torch / torchvision / pillow, none of which the service uses. The model server is
# now the shared FastAPI app (serving/app.py, ONNX Runtime) on port 8080; `main.py` is the Vast.ai PyWorker in
# front of it.
FROM vastai/base-image:cuda-12.8.1-auto

COPY requirements-serving.txt /opt/strnadi/requirements-serving.txt
RUN . /venv/main/bin/activate && pip install --no-cache-dir -r /opt/strnadi/requirements-serving.txt vastai

COPY config.py runner.py /opt/strnadi/
COPY serving /opt/strnadi/serving
COPY processing/__init__.py processing/segment.py /opt/strnadi/processing/
COPY vastai-serverless/main.py vastai-serverless/make_benchmark_samples.py vastai-serverless/start.sh /opt/strnadi/vastai/

ARG HOST_MODEL_PATH=model.onnx
ENV MODEL_PATH=/opt/strnadi/model.onnx MODEL_LOG_FILE=/var/log/model/server.log MODEL_SERVER_PORT=8080
COPY ${HOST_MODEL_PATH} ${MODEL_PATH}

RUN chmod +x /opt/strnadi/vastai/start.sh
EXPOSE 8080
ENTRYPOINT ["/opt/strnadi/vastai/start.sh"]
