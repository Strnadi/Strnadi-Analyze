from pathlib import Path
from collections import OrderedDict
from birdnet import SpeciesPredictions, predict_species_within_audio_file

YELLOWHAMMER_ID = 'Emberiza citrinella_Yellowhammer'
MIN_CONFIDENCE_TRESHOLD = 0.4
OVERLAP_TRESH = 1.5


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
    # predict species within the whole audio file
    audio_path = Path(path)

    # OrderedDict[Tuple[float, float], OrderedDict[str, float]]
    predictions = SpeciesPredictions(predict_species_within_audio_file( audio_path, 
        min_confidence=MIN_CONFIDENCE_TRESHOLD,
        chunk_overlap_s=OVERLAP_TRESH,
        species_filter={yellowhammer_id}
    ))

    # print(predictions)

    # (start, end, confidence)
    yellowhammers :list[tuple[float, float, float]] = []

    for start_end in predictions:
        detected = predictions[start_end]
        if(yellowhammer_id not in detected): continue
        confidence = detected[yellowhammer_id]

        yellowhammers.append(
            (*start_end, confidence)
        )

    merged_yellowhammers = merge_overlaps(yellowhammers)


    # print(yellowhammers)
    return list(map(
        lambda x: (x[0], x[1]),
        merged_yellowhammers
    ))



if __name__ == "__main__":
    print(
        get_yellowhammers('../.tstdata/F002413.wav')
    )
    