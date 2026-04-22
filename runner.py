import os, io
import time
import logging
import librosa
import numpy as np

from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)


app = FastAPI()
logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
MIN_CONFIDENCE_PERCENT = float(os.environ.get("MIN_CONFIDENCE_PERCENT", 70))
MIN_REPRESENTANT_CONFIDENCE_PERCENT = float(os.environ.get("MIN_REPRESENTANT_CONFIDENCE_PERCENT", 70))
MODEL_PATH=os.environ.get("MODEL_PATH", "/app/model.tflite")
MODEL_LABELS=os.environ.get("MODEL_LABELS", "BC,BE,BhBl,BlBh,None,Unfinished,XB").split(",")


def load_audio(file: str | io.BytesIO, target_sr=SAMPLE_RATE):
    audio, sr = librosa.load(file, sr=target_sr, mono=True)
    return audio, sr

def chunk_audio(audio, clip_length=4.0, step=0.5, target_sr=SAMPLE_RATE):
    for i in range(0, len(audio), int(step * target_sr)):
        chunk = audio[i:i + int(clip_length * target_sr)]

        if len(chunk) < int(clip_length * target_sr):
            padding = int(clip_length * target_sr) - len(chunk)
            chunk = np.pad(chunk, (0, padding), 'constant')

        start_s = i / target_sr
        end_s = start_s + clip_length
        yield chunk, start_s, end_s

num_threads = None
num_batch = None
interpreter = None
input_details = None
output_details = None

def process_audio(audio, batch_size=8, thread_count=8):
    global interpreter, input_details, output_details, num_threads, num_batch

    # Initialize the interpreter just once
    if interpreter is None or num_threads != thread_count or num_batch != batch_size:
        num_batch = batch_size
        num_threads = thread_count
        interpreter = Interpreter(model_path=MODEL_PATH, num_threads=num_threads)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.resize_tensor_input(input_details[0]['index'], [num_batch, 192000])
        interpreter.allocate_tensors()

    prediction : tuple[float, float, str, float] = [] # [(start, end, label, confidence)]

    # Collect chunks into batches
    batch_chunks = []
    batch_times = []

    for chunk, start, end in chunk_audio(audio):
        batch_chunks.append(chunk)
        batch_times.append((start, end))

        if len(batch_chunks) == batch_size:
            # Process full batch
            interpreter.set_tensor(input_details[0]['index'], np.array(batch_chunks))
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            for i, (start, end) in enumerate(batch_times):
                predicted = zip(MODEL_LABELS, output_data[i])
                best_label, best_confidence = max(predicted, key=lambda x: x[1])
                if best_label != 'None' and best_confidence > (MIN_CONFIDENCE_PERCENT / 100.0):
                    prediction.append((start, end, best_label, float(best_confidence), dict(predicted)))
                elif best_label != 'None':
                    prediction.append((start, end, "Unknown", float(best_confidence), dict(predicted)))

            batch_chunks = []
            batch_times = []

    # Process remaining chunks (partial batch)
    if batch_chunks:
        # Pad batch to full size with zeros
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
            predicted = zip(MODEL_LABELS, output_data[i])
            best_label, best_confidence = max(predicted, key=lambda x: x[1])
            if best_label != 'None' and best_confidence > (MIN_CONFIDENCE_PERCENT / 100.0):
                prediction.append((start, end, best_label, float(best_confidence), dict(predicted)))
            elif best_label != 'None':
                prediction.append((start, end, "Unknown", float(best_confidence), dict(predicted)))

    return prediction

def merge_overlaps_simple(detections: list[tuple[float, float, str, float, dict]], FALL_THRESHOLD=0.8) -> list[tuple[float, float, str, float, dict]]:
    """
    Merge overlapping detections (start, end, dialect, confidence).
    Keeps the most confident one if they overlap above threshold.
    """
    detections.sort(key=lambda x: x[0])
    merged = []

    for det in detections:
        if not merged:
            merged.append(det)
            continue

        last = merged[-1]

        if(last[1] + 1 > det[0]):
            # Merge them — keep the one with higher confidence
            if det[3] > last[3]:
                merged[-1] = det
            else:   
                if abs(det[3] - last[3]) >= FALL_THRESHOLD:
                    merged.append(det)
        else:
            merged.append(det)

    return merged

@app.post("/classify")
async def process(file: UploadFile):
    audio, sr = load_audio(file.file)
    logger.info(f"Loaded audio {len(audio) / sr:.2f} seconds long")

    start = time.time()
    raw_segments = process_audio(audio, thread_count=8, batch_size=8)
    merged_segments = merge_overlaps_simple(raw_segments)
    end = time.time()

    logger.info(f"Processing took {end - start:.2f} seconds")

    return JSONResponse({
        "representantId": 0 if len(merged_segments) > 0 else -1,  #"Deprecated, use 'isRepresentant' field in 'segments'. Multiple representants are allowed",
        "segments": [
            {
                "interval": [pred_start, pred_end],
                "label": label,
                "isRepresentant": confidence >= MIN_CONFIDENCE_PERCENT,
                "fullPredictions": pred_percents
            }
            for pred_start, pred_end, label, confidence, pred_percents in merged_segments
            if label != "None"
        ]
    })
