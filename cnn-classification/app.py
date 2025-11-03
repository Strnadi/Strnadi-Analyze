from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, wave, shutil
import io, zipfile, os
import numpy as np

from preprocess.preprocess import preprocess_file
from model.load import load_model

LABELS = ['3S', 'BBe', 'BC', 'BD', 'BE', 'BhBl', 'BlBh', 'None', 'Unfinished', 'XlB', 'XsB']
# LABELS = ['3S', 'BC', 'BD', 'BE', 'BhBl', 'BlBh', 'XlB', 'XsB']


app = FastAPI()

model = load_model(os.environ.get("MODEL_PATH"))


@app.post("/classify")
async def process(file: UploadFile):
    # TODO: use in-memory file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)


    segments = preprocess_file(tmp_path)

    x = np.stack(
        list(map(lambda x: x[1], segments))
    ).astype(np.float32)

    predictions = model.predict(x)

    predictions_response = []
    for i in range(len(segments)):
        prediction = predictions[i]
        interval, _ = segments[i]
        pred_percents = list(zip(LABELS, map(lambda x: round(float(x), 2) * 100, prediction.flatten())))

        predictions_response.append(
            {
                "interval": interval,
                "label": max(pred_percents, key=lambda x: x[1])[0],
                "full_prediction": dict(pred_percents)
            }
        )

    return JSONResponse(content=predictions_response)
    



