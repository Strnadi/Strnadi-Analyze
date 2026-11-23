"""Classification service. Same `/classify` JSON contract as the previous TFLite runner:

    {"representantId": int,                       # index into segments, -1 if none (deprecated, kept)
     "segments": [{"interval": [start_s, end_s], "label": str, "isRepresentant": bool,
                   "fullPredictions": {label: probability}}]}

POST /classify        multipart upload (production)
POST /classify-json   {"audio_b64": "<base64 audio file>", "filename": "..."} (Vast.ai benchmark path)
GET  /health

Fixed relative to the old runner: `fullPredictions` was always `{}` (a `zip` iterator had already been consumed
by `max`), a recording with no detections raised IndexError, and intervals were arbitrary 4 s windows rather than
songs. Environment: MODEL_PATH, MODEL_LABELS, MIN_CONFIDENCE_PERCENT, MIN_REPRESENTANT_CONFIDENCE_PERCENT,
THREADS, MODEL_LOG_FILE (optional second log target, tailed by the Vast.ai worker).
"""

import base64
import io
import logging
import os
import time

import librosa
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from serving.pipeline import SongClassifier, SR

handlers = [logging.StreamHandler()]
if os.environ.get('MODEL_LOG_FILE'):
    os.makedirs(os.path.dirname(os.environ['MODEL_LOG_FILE']), exist_ok=True)
    handlers.append(logging.FileHandler(os.environ['MODEL_LOG_FILE']))
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO,
    handlers=handlers,
)
log = logging.getLogger('strnadi')

app = FastAPI()
_classifier = None


def classifier():
    global _classifier
    if _classifier is None:
        _classifier = SongClassifier(
            os.environ.get('MODEL_PATH', '/app/model.onnx'),
            os.environ.get('MODEL_LABELS', 'BC,BE,BhBl,BlBh,None,Unfinished,XB').split(','),
            threads=int(os.environ.get('THREADS', 4)),
            min_confidence=float(os.environ.get('MIN_CONFIDENCE_PERCENT', 60)) / 100,
            min_representant=float(os.environ.get('MIN_REPRESENTANT_CONFIDENCE_PERCENT', 65)) / 100,
        )
    return _classifier


@app.on_event('startup')
async def startup():
    classifier()
    log.info('Application startup complete.')


def run(data: bytes):
    try:
        y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'could not decode audio: {e}')
    t = time.time()
    out = classifier().classify(y)
    log.info(f'{len(y) / SR:.1f} s audio, {len(out["segments"])} segments, {time.time() - t:.2f} s')
    return JSONResponse(out)


@app.post('/classify')
async def classify(file: UploadFile):
    return run(await file.read())


class JsonAudio(BaseModel):
    audio_b64: str
    filename: str | None = None


@app.post('/classify-json')
async def classify_json(body: JsonAudio):
    try:
        data = base64.b64decode(body.audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail='audio_b64 is not valid base64')
    return run(data)


@app.get('/health')
async def health():
    return {'status': 'ok'}
