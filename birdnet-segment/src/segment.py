from pathlib import Path
import birdnet
import os, time
import logging
from multiprocessing import cpu_count

YELLOWHAMMER_ID = 'Emberiza citrinella_Yellowhammer'
MIN_CONFIDENCE_TRESHOLD = float(os.environ.get("BIRDNET_MIN_CONFIDENCE_TRESH", 0.4))
OVERLAP_TRESH = 2.5
# OVERLAP_TRESH = 1

logger = logging.getLogger("segment")

workers = cpu_count() - 1
if(workers <= 0): workers = 1

logger.info("initializing birdnet model")
model = birdnet.load("acoustic", "2.4", "tf")
logger.info("birdnet model initialized")


FALL_TRESHOLD = 0.8

# '''
def merge_overlaps(detections: list[tuple[float, float, float]]):
    """
    Merge overlapping detections (start, end, confidence).
    Keeps the most confident one if they overlap above threshold.
    """
    detections.sort(key=lambda x: x[0])
    merged = []

    # ceiling = 

    for det in detections:
        if not merged:
            merged.append(det)
            continue

        last = merged[-1]

        if(last[1] + 1 > det[0]):
            # Merge them — keep the one with higher confidence
            if det[2] > last[2]:
                merged[-1] = det
            else:
                if abs(det[2] - last[2]) >= FALL_TRESHOLD:
                    merged.append(det)
        else:
            merged.append(det)

    return merged
# '''
'''
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

        if(last[1] + 1 > det[0]):
            merged[-1] = [
                min(last[0], det[0]),
                max(last[1], det[1]),
                (last[2] + det[2]) / 2
            ]
        else:
            mid = merged[-1][0] + (abs(merged[-1][0] - merged[-1][1]) / 2)
            merged[-1] = (mid - 1, mid + 2, merged[-1][2])

            merged.append(det)

    if abs(merged[-1][0] - merged[-1][1]) != 4:
        mid = merged[-1][0] + (abs(merged[-1][0] - merged[-1][1]) / 2)
        merged[-1] = (mid - 1, mid + 2, merged[-1][2])

    return merged
'''


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
        batch_size=1,
        custom_species_list={yellowhammer_id},
        overlap_duration_s=OVERLAP_TRESH,
        n_workers= workers
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

    logger.debug(f"direct birdnet predictions: {yellowhammers}")
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
    