"""
Model server — runs on localhost:8080.
PyWorker (worker.py) proxies inbound traffic to this process.

Log output is written to MODEL_LOG_FILE so that the PyWorker can tail it
and detect the "Application startup complete." readiness signal.

Endpoints
─────────
POST /classify          Multipart upload (UploadFile). Production path.
POST /classify-json     JSON body {audio_b64, filename?}. Used by PyWorker
                        benchmarks — BenchmarkConfig.generator must return
                        a JSON-serialisable dict, so multipart is not an
                        option for the benchmark route.
GET  /health            Lightweight liveness check.
"""

import base64
import io
import logging
import os
import time

import librosa
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging — dual output so worker.py can tail the file for readiness detection
# ---------------------------------------------------------------------------

LOG_FILE = os.environ.get("MODEL_LOG_FILE", "/var/log/model/server.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)

# ---------------------------------------------------------------------------
# Constants / environment
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000
MIN_CONFIDENCE_PERCENT = float(os.environ.get("MIN_CONFIDENCE_PERCENT", 70))
MIN_REPRESENTANT_CONFIDENCE_PERCENT = float(
    os.environ.get("MIN_REPRESENTANT_CONFIDENCE_PERCENT", 70)
)
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model.tflite")
MODEL_LABELS = os.environ.get(
    "MODEL_LABELS", "BC,BE,BhBl,BlBh,None,Unfinished,XB"
).split(",")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup: emit the exact log line that worker.py watches for
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    logger.info("Application startup complete.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(file: str | io.BytesIO, target_sr: int = SAMPLE_RATE):
    audio, sr = librosa.load(file, sr=target_sr, mono=True)
    return audio, sr


def chunk_audio(
    audio: np.ndarray,
    clip_length: float = 4.0,
    step: float = 0.5,
    target_sr: int = SAMPLE_RATE,
):
    clip_samples = int(clip_length * target_sr)
    step_samples = int(step * target_sr)

    for i in range(0, len(audio), step_samples):
        chunk = audio[i : i + clip_samples]
        if len(chunk) < clip_samples:
            chunk = np.pad(chunk, (0, clip_samples - len(chunk)), "constant")
        yield chunk, i / target_sr, i / target_sr + clip_length


# ---------------------------------------------------------------------------
# TFLite interpreter (lazy singleton — not thread-safe, serialised by worker.py)
# ---------------------------------------------------------------------------

_interp_state: dict = {
    "interpreter":   None,
    "input_details":  None,
    "output_details": None,
    "num_threads":    None,
    "num_batch":      None,
}


def _ensure_interpreter(batch_size: int, thread_count: int) -> None:
    s = _interp_state
    if (
        s["interpreter"] is None
        or s["num_threads"] != thread_count
        or s["num_batch"] != batch_size
    ):
        s["num_batch"] = batch_size
        s["num_threads"] = thread_count
        interp = Interpreter(model_path=MODEL_PATH, num_threads=thread_count)
        inp = interp.get_input_details()
        out = interp.get_output_details()
        interp.resize_tensor_input(inp[0]["index"], [batch_size, 192000])
        interp.allocate_tensors()
        s["interpreter"] = interp
        s["input_details"] = inp
        s["output_details"] = out
        logger.info(
            "TFLite interpreter initialised "
            f"(threads={thread_count}, batch={batch_size})"
        )


def _run_batch(
    batch_chunks: list[np.ndarray],
    batch_times: list[tuple[float, float]],
) -> list[tuple[float, float, str, float, dict]]:
    s = _interp_state
    s["interpreter"].set_tensor(
        s["input_details"][0]["index"],
        np.array(batch_chunks, dtype=np.float32),
    )
    s["interpreter"].invoke()
    output_data = s["interpreter"].get_tensor(s["output_details"][0]["index"])

    results = []
    for i, (start, end) in enumerate(batch_times):
        pairs = list(zip(MODEL_LABELS, output_data[i]))
        best_label, best_conf = max(pairs, key=lambda x: x[1])
        if best_label == "None":
            continue
        label = best_label if best_conf >= MIN_CONFIDENCE_PERCENT / 100.0 else "Unknown"
        results.append((start, end, label, float(best_conf), dict(pairs)))
    return results


def process_audio(
    audio: np.ndarray,
    batch_size: int = 8,
    thread_count: int = 8,
) -> list[tuple[float, float, str, float, dict]]:
    _ensure_interpreter(batch_size, thread_count)

    predictions: list[tuple[float, float, str, float, dict]] = []
    batch_chunks: list[np.ndarray] = []
    batch_times: list[tuple[float, float]] = []

    for chunk, start, end in chunk_audio(audio):
        batch_chunks.append(chunk)
        batch_times.append((start, end))

        if len(batch_chunks) == batch_size:
            predictions.extend(_run_batch(batch_chunks, batch_times))
            batch_chunks = []
            batch_times = []

    # Partial batch — pad with silence, only use real results
    if batch_chunks:
        real_count = len(batch_chunks)
        while len(batch_chunks) < batch_size:
            batch_chunks.append(np.zeros(192000, dtype=np.float32))
            batch_times.append((0.0, 0.0))
        predictions.extend(_run_batch(batch_chunks, batch_times)[:real_count])

    return predictions


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def merge_overlaps_simple(
    detections: list[tuple[float, float, str, float, dict]],
    fall_threshold: float = 0.8,
) -> list[tuple[float, float, str, float, dict]]:
    detections.sort(key=lambda x: x[0])
    merged: list[tuple[float, float, str, float, dict]] = []

    for det in detections:
        if not merged:
            merged.append(det)
            continue
        last = merged[-1]
        if last[1] + 1 > det[0]:
            if det[3] > last[3]:
                merged[-1] = det
            elif abs(det[3] - last[3]) >= fall_threshold:
                merged.append(det)
        else:
            merged.append(det)

    return merged


def _build_response(merged: list[tuple[float, float, str, float, dict]]) -> dict:
    return {
        "representantId": (
            "Deprecated, use 'isRepresentant' field in 'segments'. "
            "Multiple representants are allowed"
        ),
        "segments": [
            {
                "interval": [pred_start, pred_end],
                "label": label,
                "isRepresentant": confidence >= MIN_REPRESENTANT_CONFIDENCE_PERCENT / 100.0,
                "fullPredictions": pred_percents,
            }
            for pred_start, pred_end, label, confidence, pred_percents in merged
            if label != "None"
        ],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Lightweight liveness check."""
    return {"status": "ok"}


@app.post("/classify")
async def classify_upload(file: UploadFile):
    """Production endpoint — multipart WAV/audio upload."""
    audio, sr = load_audio(file.file)
    logger.info(f"[classify] loaded {len(audio)/sr:.2f}s audio")

    t0 = time.time()
    raw = process_audio(audio, thread_count=8, batch_size=8)
    merged = merge_overlaps_simple(raw)
    logger.info(f"[classify] processed in {time.time() - t0:.2f}s")

    return JSONResponse(_build_response(merged))


# ---------------------------------------------------------------------------
# Request model for the benchmark / JSON endpoint
# ---------------------------------------------------------------------------

class ClassifyJsonRequest(BaseModel):
    """
    audio_b64 : base64-encoded WAV file (any sample rate; librosa resamples)
    filename  : optional label used only for logging
    """
    audio_b64: str
    filename: str = "benchmark_sample"


@app.post("/classify-json")
async def classify_json(req: ClassifyJsonRequest):
    """
    Benchmark-friendly counterpart to /classify.

    Accepts a JSON body instead of a multipart upload so that
    BenchmarkConfig.generator (which must return a plain dict) can drive it.
    The payload shape mirrors what benchmark_samples.random_sample() returns:

        {"audio_b64": "<base64-encoded WAV>", "filename": "<label>"}
    """
    try:
        raw_bytes = base64.b64decode(req.audio_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    audio, sr = load_audio(io.BytesIO(raw_bytes))
    logger.info(f"[classify-json] {req.filename}: {len(audio)/sr:.2f}s")

    t0 = time.time()
    raw = process_audio(audio, thread_count=8, batch_size=8)
    merged = merge_overlaps_simple(raw)
    logger.info(f"[classify-json] {req.filename}: processed in {time.time() - t0:.2f}s")

    return JSONResponse(_build_response(merged))
