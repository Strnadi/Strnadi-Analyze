from pathlib import Path
from collections import OrderedDict
from birdnet import SpeciesPredictions, predict_species_within_audio_file

YELLOWHAMMER_ID = 'Emberiza citrinella_Yellowhammer'
MIN_CONFIDENCE_TRESHOLD = 0.2


def get_yellowhammers(path: str, min_confidence_tresh=MIN_CONFIDENCE_TRESHOLD, yellowhammer_id=YELLOWHAMMER_ID) -> list[tuple[float, float, float]]  :
    # predict species within the whole audio file
    audio_path = Path(path)

    # OrderedDict[Tuple[float, float], OrderedDict[str, float]]
    predictions = SpeciesPredictions(predict_species_within_audio_file(audio_path))



    # (start, end, confidence)
    yellowhammers :list[tuple[float, float, float]] = []

    for start_end in predictions:
        detected = predictions[start_end]
        if(YELLOWHAMMER_ID in detected):
            
            if(detected[YELLOWHAMMER_ID] < MIN_CONFIDENCE_TRESHOLD): continue

            yellowhammers.append(
                start_end
            )

    # print(yellowhammers)
    return yellowhammers

