from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, wave, shutil
import io, zipfile, os
import numpy as np

from preprocess.preprocess import preprocess
from model.load import load_model

LABELS = ['3S', 'BBe', 'BC', 'BD', 'BE', 'BhBl', 'BlBh', 'None', 'Unfinished', 'XlB', 'XsB']
# LABELS = ['3S', 'BC', 'BD', 'BE', 'BhBl', 'BlBh', 'XlB', 'XsB']

MIN_CONFIDENCE_PERCENT = float(os.environ.get("MIN_CONFIDENCE_PERCENT", 70))
MIN_REPRESENTANT_CONFIDENCE_PERCENT = float(os.environ.get("MIN_REPRESENTANT_CONFIDENCE_PERCENT", 70))

UNKNOWN = "Unknown"

app = FastAPI()

model = load_model(os.environ.get("MODEL_PATH"))


@app.post("/classify")
async def process(file: UploadFile):

    segments = preprocess(file.file.read())


    if len(segments) == 0: # no segments => do early return
            return JSONResponse(content={
                "representant": "None",
                "confidence": 100,
                "segments": []
            }
        )

    x = np.stack(
        list(map(lambda x: x[1], segments))
    ).astype(np.float32)

    predictions = model.predict(x)
    

    prediction_sum = np.zeros(len(LABELS), np.float32)

    # ==== per segment prediction =====
    segments_response = []
    for i in range(len(segments)):
        prediction = predictions[i]
        interval, _ = segments[i]
        pred_percents = list(zip(LABELS, map(lambda x: round(float(x), 2) * 100, prediction.flatten())))
        most_probable_pred = max(pred_percents, key=lambda x: x[1])

        if most_probable_pred[0] == 'None':
            continue # if segment has None yellowhammers, there's no point in returning it

        segments_response.append(
            {
                "interval": interval,
                "label": most_probable_pred[0] if most_probable_pred[1] >= MIN_CONFIDENCE_PERCENT else UNKNOWN,
                "full_prediction": dict(pred_percents)
            }
        )

        prediction_sum += prediction
    

    most_probable_representant = ("None", 100)

    # if all segments are None, this if will get skipped (default to "None")
    if len(segments_response) != 0:
        # ==== compute overall prediction summary ====
        prediction_avg = prediction_sum / len(segments_response)
        representant_pred_percent = list(zip(LABELS, map(lambda x: round(float(x), 2) * 100, prediction_avg)))

        most_probable_representant = max(representant_pred_percent, key=lambda x: x[1])
        if most_probable_representant[1] < MIN_REPRESENTANT_CONFIDENCE_PERCENT:
            most_probable_representant = (UNKNOWN, None)


    return JSONResponse(content={
            "representant": most_probable_representant[0],
            "confidence": most_probable_representant[1],
            "segments": segments_response
        }
    )
    



