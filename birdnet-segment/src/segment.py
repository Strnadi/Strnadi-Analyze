from pathlib import Path
import birdnet
import os, time
import logging

YELLOWHAMMER_ID = 'Emberiza citrinella_Yellowhammer'
MIN_CONFIDENCE_TRESHOLD = float(os.environ.get("BIRDNET_MIN_CONFIDENCE_TRESH", 0.4))
OVERLAP_TRESH = 2.5

logger = logging.getLogger("segment")

logger.info("initializing birdnet model")
model = birdnet.load("acoustic", "2.4", "tf")
logger.info("birdnet model initialized")

def merge_overlaps(detections: list[tuple[float, float, float]]):
    """
    Merge overlapping detections (start, end, confidence).
    Keeps the most confident one if they overlap above threshold.
    """
    detections.sort(key=lambda x: x[0])
    merged = []

    for det in detections:
        if not merged:
            merged.append(det)
            continue

        last = merged[-1]

        if(last[1] > det[0]):
            # Merge them — keep the one with higher confidence
            if det[2] > last[2]:
                merged[-1] = det
            else:
                pass
                # merged.append(det)
        else:
            merged.append(det)

    return merged



def get_yellowhammers(path: str, min_confidence_tresh=MIN_CONFIDENCE_TRESHOLD, yellowhammer_id=YELLOWHAMMER_ID) -> list[tuple[float, float, float]]  :
    global model
    # predict species within the whole audio file
    audio_path = Path(path)

    logger.info("starting birdnet prediction")
    s_t = time.time()
    # OrderedDict[Tuple[float, float], OrderedDict[str, float]]
    predictions_obj = model.predict(
        audio_path,
        default_confidence_threshold=MIN_CONFIDENCE_TRESHOLD,
        batch_size=16,
        custom_species_list={yellowhammer_id},
        # species_filter={yellowhammer_id}
    )

    d_t = time.time() - s_t
    logger.info(f"birdnet prediction took {d_t} seconds")

    predictions = predictions_obj.to_structured_array()

    # (start, end, confidence)
    yellowhammers :list[tuple[float, float, float]] = []

    for _, start, end, species, confidence in predictions:

        yellowhammers.append(
            (float(start), float(end), float(confidence))
        )

    logger.debug("merging overlaps")
    merged_yellowhammers = merge_overlaps(yellowhammers)


    # print(yellowhammers)
    return list(map(
        lambda x: (x[0], x[1]),
        merged_yellowhammers
    ))



# if __name__ == "__main__":
#     print(
#         get_yellowhammers('../.tstdata/F002413.wav')
#     )
    