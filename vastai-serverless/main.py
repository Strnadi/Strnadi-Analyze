"""
worker.py — Vast.ai Serverless PyWorker

Architecture
────────────
  Vast router  ──►  PyWorker (this file, port 8000)  ──►  FastAPI (server.py, port 8080)

Benchmark design
────────────────
  BenchmarkConfig.generator must return a plain JSON-serialisable dict.
  /classify requires multipart upload, which the benchmark machinery cannot
  produce.  Solution: server.py exposes /classify-json that accepts
  {"audio_b64": "<base64 WAV>", "filename": "<label>"}.

  benchmark_samples.py (auto-generated) embeds three representative WAV files:
    • short_tone   4 s  clean 440 Hz sine
    • multi_tone   8 s  overlapping harmonics with FM modulation
    • noise_burst  4 s  band-limited noise 200–4000 Hz

  The benchmark generator picks one sample at random per call, so the
  PyWorker benchmarks with varied inputs that represent the real workload
  distribution.

  Workload is measured in seconds of audio so the autoscaler's throughput
  estimate (workload / second) maps to a tangible unit: "audio-seconds per
  wall-second".
"""

import os

from aiohttp import ClientResponse, web

from vastai import (
    BenchmarkConfig,
    HandlerConfig,
    LogActionConfig,
    Worker,
    WorkerConfig,
)

from benchmark_samples import random_sample

# ---------------------------------------------------------------------------
# Model server coordinates
# ---------------------------------------------------------------------------

MODEL_SERVER_URL  = "http://127.0.0.1"
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", 8080))
MODEL_LOG_FILE    = os.environ.get("MODEL_LOG_FILE", "/var/log/model/server.log")

# ---------------------------------------------------------------------------
# Response generator
#
# Both /classify and /classify-json return application/json.  We echo the
# body and status unchanged; the only customisation is surfacing the model
# server's status code faithfully (e.g. 400 for malformed audio).
# ---------------------------------------------------------------------------

async def json_passthrough(
    client_request: web.Request,
    model_response: ClientResponse,
) -> web.Response:
    body = await model_response.read()
    return web.Response(
        body=body,
        status=model_response.status,
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Benchmark generator
#
# Called once per benchmark run by PyWorker.  Returns a dict that is
# JSON-POSTed to /classify-json on the model server.
#
# Workload = duration of the audio clip in seconds.  This gives the
# autoscaler a meaningful unit: "audio-seconds processed per wall-second".
# ---------------------------------------------------------------------------

# Pre-decode durations so we can return exact workloads without re-parsing.
import base64, io, struct

def _wav_duration_seconds(b64_data: str) -> float:
    """Read sample-count and sample-rate from a WAV header without scipy."""
    raw = base64.b64decode(b64_data)
    buf = io.BytesIO(raw)
    buf.seek(24)                       # byte offset of sample rate field
    sample_rate = struct.unpack_from("<I", buf.read(4))[0]
    buf.seek(4)
    chunk_size  = struct.unpack_from("<I", buf.read(4))[0]
    # chunk_size = file_size - 8; data payload starts at byte 44 for PCM WAV
    data_bytes  = chunk_size - 36
    buf.seek(34)
    bits_per_sample = struct.unpack_from("<H", buf.read(2))[0]
    buf.seek(22)
    num_channels = struct.unpack_from("<H", buf.read(2))[0]
    bytes_per_sample = bits_per_sample // 8
    num_samples = data_bytes // (bytes_per_sample * num_channels)
    return num_samples / sample_rate


from benchmark_samples import SAMPLES as _SAMPLES

_SAMPLE_DURATIONS: dict[str, float] = {
    name: _wav_duration_seconds(b64)
    for name, b64 in _SAMPLES.items()
}


def classify_benchmark_generator() -> dict:
    """
    Return one benchmark payload for POST /classify-json.

    Picks a random sample each call so the benchmark exercises a range of
    audio lengths and content types.  The chosen filename is included purely
    for logging on the server side.
    """
    name, b64_data = random_sample()
    return {
        "audio_b64": b64_data,
        "filename": name,
    }


def classify_workload_calculator(payload: dict) -> float:
    """
    Return the audio duration in seconds.

    For benchmark payloads the filename is always a known key; for
    production /classify-json calls the filename is arbitrary so we fall
    back to decoding the WAV header on the fly.
    """
    filename = payload.get("filename", "")
    if filename in _SAMPLE_DURATIONS:
        return _SAMPLE_DURATIONS[filename]

    # Production path: decode the header from the actual payload
    b64_data = payload.get("audio_b64", "")
    if b64_data:
        try:
            return _wav_duration_seconds(b64_data)
        except Exception:
            pass

    return 4.0   # safe fallback: assume one clip length


# ---------------------------------------------------------------------------
# WorkerConfig
# ---------------------------------------------------------------------------

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,

    handlers=[
        # ── /classify-json ────────────────────────────────────────────────
        # JSON variant — used by benchmarks and any caller that prefers
        # not to send multipart.  FIFO-queued (same TFLite singleton as
        # /classify).  This is the BENCHMARK HANDLER.
        HandlerConfig(
            route="/classify-json",
            allow_parallel_requests=False,  # TFLite interpreter is not reentrant
            max_queue_time=120.0,
            workload_calculator=classify_workload_calculator,
            response_generator=json_passthrough,
            benchmark_config=BenchmarkConfig(
                # generator returns real audio samples — representative of
                # the actual workload the model will run in production.
                generator=classify_benchmark_generator,
                runs=6,         # 2 runs × 3 samples → covers all sample types
                concurrency=1,  # must be 1: TFLite runs serially
            ),
        ),

        # ── /classify ────────────────────────────────────────────────────
        # Production multipart-upload path.  Shares the same TFLite
        # singleton with /classify-json, so FIFO queueing is essential.
        # workload_calculator cannot inspect multipart body easily, so we
        # use a constant representing one "typical" 4-second clip.
        HandlerConfig(
            route="/classify",
            allow_parallel_requests=False,
            max_queue_time=120.0,
            workload_calculator=lambda _: 4.0,  # one clip ≈ 4 s of audio
            response_generator=json_passthrough,
            # No benchmark_config: multipart payloads can't be generated
            # by BenchmarkConfig.generator.
        ),

        # ── /health ──────────────────────────────────────────────────────
        # Liveness probe only; not used for benchmarking.
        HandlerConfig(
            route="/health",
            allow_parallel_requests=True,
            max_queue_time=5.0,
            workload_calculator=lambda _: 0.0,
            response_generator=json_passthrough,
        ),
    ],

    log_action_config=LogActionConfig(
        on_load=[
            # Emitted by server.py's @app.on_event("startup") callback.
            # PyWorker transitions to "ready" and starts benchmarks on first match.
            "Application startup complete.",
        ],
        on_error=[
            "Traceback (most recent call last):",
            "ERROR:    Application startup failed.",
            "RuntimeError:",
        ],
        on_info=[
            "INFO:     ",
        ],
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

Worker(worker_config).run()
