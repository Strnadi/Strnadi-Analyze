#!/bin/bash
# Start the model server (FastAPI, port $MODEL_SERVER_PORT) and the Vast.ai PyWorker in front of it.
set -e
. /venv/main/bin/activate
mkdir -p "$(dirname "$MODEL_LOG_FILE")"

cd /opt/strnadi/vastai
[ -f benchmark_samples.py ] || python make_benchmark_samples.py      # synthetic clips the worker benchmarks with

cd /opt/strnadi
uvicorn serving.app:app --host 127.0.0.1 --port "${MODEL_SERVER_PORT:-8080}" &

cd /opt/strnadi/vastai
exec python main.py
