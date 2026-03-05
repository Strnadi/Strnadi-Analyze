# worker.py
# PyWorker entrypoint for Vast.ai Serverless
# - starts a local FastAPI model server (the same /classify endpoint you had)
# - registers a Handler route with the Vast Worker so Serverless can route to it
#
# Notes:
# - This expects the same dependencies as your original runner (librosa, numpy, ai_edge_litert, tflite runtime, etc.)
# - The Vast Worker will proxy incoming requests to the local model server started here.

import os
import io
import time
import logging
import threading
from typing import List, Tuple, Optional

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Vast SDK imports (install vastai-sdk / vastai)
try:
    from vastai import (
        Worker,
        WorkerConfig,
        HandlerConfig,
        BenchmarkConfig,
        LogActionConfig,
    )
except Exception as e:
    # If the package name on your system differs, install 'vastai-sdk' (pip) in requirements.txt
    raise

# your model interpreter
from ai_edge_litert.interpreter import Interpreter

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("pyworker")

# -------------------------
# Configuration (env)
# -------------------------
SAMPLE_RATE = 48000
MIN_CONFIDENCE_PERCENT = float(os.environ.get("MIN_CONFIDENCE_PERCENT", 70))
MIN_REPRESENTANT_CONFIDENCE_PERCENT = float(os.environ.get("MIN_REPRESENTANT_CONFIDENCE_PERCENT", 70))
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model.tflite")
MODEL_LABELS = os.environ.get("MODEL_LABELS", "BC,BE,BhBl,BlBh,None,Unfinished,XB").split(",")

# Local model server config (FastAPI) port — the PyWorker will proxy to this
MODEL_SERVER_HOST = os.environ.get("MODEL_SERVER_HOST", "127.0.0.1")
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "18001"))

# PyWorker log detection: mark model ready when this string appears in logs.
PYWORKER_READY_STRING = os.environ.get("PYWORKER_READY_STRING", "Application startup complete.")

# Batch/thread defaults
DEFAULT_BATCH_SIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", 8))
DEFAULT_THREAD_COUNT = int(os.environ.get("DEFAULT_THREAD_COUNT", 8))

# -------------------------
# FastAPI model server (keeps your original logic)
# -------------------------
app = FastAPI()

def load_audio(file: io.BufferedReader | str, target_sr=SAMPLE_RATE):
    audio, sr = librosa.load(file, sr=target_sr, mono=True)
    return audio, sr

def chunk_audio(audio: np.ndarray, clip_length=4.0, step=0.5, target_sr=SAMPLE_RATE):
    for i in range(0, len(audio), int(step * target_sr)):
        chunk = audio[i:i + int(clip_length * target_sr)]

        if len(chunk) < int(clip_length * target_sr):
            padding = int(clip_length * target_sr) - len(chunk)
            chunk = np.pad(chunk, (0, padding), 'constant')

        start_s = i / target_sr
        end_s = start_s + clip_length
        yield chunk.astype(np.float32), start_s, end_s

# Interpreter and caches (module-global to reuse between requests)
num_threads = None
num_batch = None
interpreter: Optional[Interpreter] = None
input_details = None
output_details = None

def process_audio(audio: np.ndarray, batch_size=DEFAULT_BATCH_SIZE, thread_count=DEFAULT_THREAD_COUNT):
    global interpreter, input_details, output_details, num_threads, num_batch

    # Initialize the interpreter just once, or if config changes
    if interpreter is None or num_threads != thread_count or num_batch != batch_size:
        num_batch = batch_size
        num_threads = thread_count
        logger.info(f"Loading Interpreter model_path={MODEL_PATH} threads={num_threads}")
        interpreter = Interpreter(model_path=MODEL_PATH, num_threads=num_threads)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # ensure input size matches [batch, 192000]
        interpreter.resize_tensor_input(input_details[0]['index'], [num_batch, 192000])
        interpreter.allocate_tensors()

    predictions = []  # list of (start, end, label, confidence, dict(preds))

    batch_chunks: List[np.ndarray] = []
    batch_times: List[Optional[Tuple[float, float]]] = []

    for chunk, start, end in chunk_audio(audio):
        batch_chunks.append(chunk)
        batch_times.append((start, end))

        if len(batch_chunks) == batch_size:
            interpreter.set_tensor(input_details[0]['index'], np.array(batch_chunks))
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            for i, times in enumerate(batch_times):
                start, end = times
                predicted = list(zip(MODEL_LABELS, output_data[i]))
                best_label, best_confidence = max(predicted, key=lambda x: x[1])
                if best_label != 'None' and best_confidence > (MIN_CONFIDENCE_PERCENT / 100.0):
                    predictions.append((start, end, best_label, float(best_confidence), dict(predicted)))
                elif best_label != 'None':
                    predictions.append((start, end, "Unknown", float(best_confidence), dict(predicted)))

            batch_chunks = []
            batch_times = []

    # remaining partial batch
    if batch_chunks:
        while len(batch_chunks) < batch_size:
            batch_chunks.append(np.zeros(192000, dtype=np.float32))
            batch_times.append(None)

        interpreter.set_tensor(input_details[0]['index'], np.array(batch_chunks))
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        for i, times in enumerate(batch_times):
            if times is None:
                break
            start, end = times
            predicted = list(zip(MODEL_LABELS, output_data[i]))
            best_label, best_confidence = max(predicted, key=lambda x: x[1])
            if best_label != 'None' and best_confidence > (MIN_CONFIDENCE_PERCENT / 100.0):
                predictions.append((start, end, best_label, float(best_confidence), dict(predicted)))
            elif best_label != 'None':
                predictions.append((start, end, "Unknown", float(best_confidence), dict(predicted)))

    return predictions

def merge_overlaps_simple(detections, FALL_THRESHOLD=0.8):
    detections.sort(key=lambda x: x[0])
    merged = []
    for det in detections:
        if not merged:
            merged.append(det)
            continue
        last = merged[-1]
        # if overlapping (last end + 1 > det start) treat as overlap
        if (last[1] + 1 > det[0]):
            if det[3] > last[3]:
                merged[-1] = det
            else:
                if abs(det[3] - last[3]) >= FALL_THRESHOLD:
                    merged.append(det)
        else:
            merged.append(det)
    return merged

@app.post("/classify")
async def classify_endpoint(file: UploadFile):
    # This endpoint is the model server's API. Vast PyWorker will proxy to it.
    # Keep behavior same as your original runner.
    logger.info("Received /classify request")
    # read file into memory (librosa accepts file-like or path)
    audio_file = file.file
    audio, sr = load_audio(audio_file)
    logger.info(f"Loaded audio {len(audio) / sr:.2f} seconds long")

    start = time.time()
    raw_segments = process_audio(audio, batch_size=DEFAULT_BATCH_SIZE, thread_count=DEFAULT_THREAD_COUNT)
    merged_segments = merge_overlaps_simple(raw_segments)
    end = time.time()
    logger.info(f"Processing took {end - start:.2f} seconds")

    segments = [
        {
            "interval": [pred_start, pred_end],
            "label": label,
            "isRepresentant": confidence >= (MIN_REPRESENTANT_CONFIDENCE_PERCENT / 100.0),
            "fullPredictions": pred_percents
        }
        for pred_start, pred_end, label, confidence, pred_percents in merged_segments
        if label != "None"
    ]

    return JSONResponse({
        "representantId": "Deprecated, use 'isRepresentant' field in 'segments'. Multiple representants are allowed",
        "segments": segments,
    })


# -------------------------
# Helper to run uvicorn in a thread
# -------------------------
def _run_model_server_in_thread():
    config = uvicorn.Config(app=app, host=MODEL_SERVER_HOST, port=MODEL_SERVER_PORT, log_level="info")
    server = uvicorn.Server(config=config)
    # run server (blocking) in a thread
    server.run()

# -------------------------
# PyWorker configuration & startup
# -------------------------
def start_pyworker():
    # Build a WorkerConfig describing the model server and exposed handlers.
    # The Worker proxies /classify to the local FastAPI server we started above.
    model_server_url = f"http://{MODEL_SERVER_HOST}"
    model_log_file = os.environ.get("MODEL_LOG_FILE", "/var/log/model/server.log")

    # A simple workload calculator. For audio classification a constant cost is reasonable;
    # you can replace with audio-length-based cost if your client sends metadata.
    workload_calculator = lambda payload: float(payload.get("work", 100.0)) if isinstance(payload, dict) else 100.0

    # Benchmark config - a minimal generator. This lets Vast estimate capacity.
    benchmark_config = BenchmarkConfig(generator=lambda: {"file_bytes": b"\x00"*1000}, runs=1, concurrency=1)


    handler = HandlerConfig(
        route="/classify",
        allow_parallel_requests=False,  # TFLite interpreter is usually single-threaded safe
        max_queue_time=60.0,
        workload_calculator=workload_calculator,
        benchmark_config=benchmark_config,
    )

    log_action_config = LogActionConfig(
        on_load=[PYWORKER_READY_STRING],  # mark ready when this string appears in logs
        on_error=["Traceback (most recent call last):", "RuntimeError:"],
        on_info=['"message":"Download']  # optional additional info triggers
    )

    worker_config = WorkerConfig(
        model_server_url=model_server_url,
        model_server_port=MODEL_SERVER_PORT,
        model_log_file=model_log_file,
        handlers=[handler],
        log_action_config=log_action_config,
    )

    logger.info("Starting Vast PyWorker with config: %s", worker_config)
    w = Worker(worker_config)
    w.run()  # this call blocks and manages the PyWorker HTTP server and autoscaling metrics

# -------------------------
# main: start model server then start pyworker loop
# -------------------------
if __name__ == "__main__":
    # Start model server in background thread
    t = threading.Thread(target=_run_model_server_in_thread, daemon=True)
    t.start()

    # Wait a touch to allow server to start and emit logs
    time.sleep(1.0)
    # Print a readiness marker that the PyWorker can detect in logs (or use model logs path).
    logger.info(PYWORKER_READY_STRING)

    # Start the Vast PyWorker control loop (this will block)
    start_pyworker()
