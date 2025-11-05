from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import os, time
import numpy as np
import logging


from preprocess.preprocess import preprocess
from model.load import load_model

LABELS = ['3S', 'BBe', 'BC', 'BD', 'BE', 'BhBl', 'BlBh', 'None', 'Unfinished', 'XlB', 'XsB']
# LABELS = ['3S', 'BC', 'BD', 'BE', 'BhBl', 'BlBh', 'XlB', 'XsB']

MIN_CONFIDENCE_PERCENT = float(os.environ.get("MIN_CONFIDENCE_PERCENT", 70))
MIN_REPRESENTANT_CONFIDENCE_PERCENT = float(os.environ.get("MIN_REPRESENTANT_CONFIDENCE_PERCENT", 70))

UNKNOWN = "Unknown"

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("app")
app = FastAPI()

logger.info("loading model")
model = load_model(os.environ.get("MODEL_PATH"))
logger.info("model loaded")


@app.post("/classify")
async def process(file: UploadFile):
    logger.info("got file to process")

    segments = preprocess(file.file.read())


    if len(segments) == 0: # no segments => do early return
            logger.info("no yellowhammers were found in recording")
            return JSONResponse(content={
                "representantId": -1,
                "segments": []
            }
        )

    x = np.stack(
        list(map(lambda x: x[1], segments))
    ).astype(np.float32)

    logger.info(f"predicting dialects for {len(segments)} segments")
    s_t = time.time()

    predictions = model.predict(x)

    d_t = time.time() - s_t
    logger.info(f"dialect prediction took {d_t} seconds")
    

    

    # ==== per segment prediction =====
    all_max_predictions = []

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
                "fullPredictions": dict(pred_percents)
            }
        )
        all_max_predictions.append(most_probable_pred[1])

    
    representant_id = -1 if len(all_max_predictions) == 0 else all_max_predictions.index(max(all_max_predictions))

    return JSONResponse(content={
            "representantId": representant_id,
            "segments": segments_response
        }
    )
    



