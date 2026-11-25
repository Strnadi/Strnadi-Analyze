"""Concept bottleneck: how close are the measurements the network reads off a held-out clip?

    python -m evaluation.measurement_bottleneck <run>

For a run of `experiments/measurement_bottleneck.py`, whose class is decided from eight
measurements of the song ending (`layers/measurement_head.py`). On the held-out clips of every
fold the predicted measurements are compared with the values measured inside the element spans
(`processing/build_measurements_from_tones.py`):

- median absolute error of the six continuous measurements (kHz; t2 in seconds), over the clips
  that have the value;
- accuracy of the two presence outputs (is there a first / a second terminal element);
- the class median of every predicted measurement next to the measured one: the table a reader
  uses to see that the bottleneck holds what its names say.

Writes `results/measurement_bottleneck_<run>.json`.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from config import CLASSES, WS
from evaluation.explain_model import FoldModels, clip_spectrogram
from evaluation.report import load_run_predictions
from layers.measurement_head import MEASUREMENTS
from training.data import read_table

CONTINUOUS = MEASUREMENTS[:6]
PRESENCE = MEASUREMENTS[6:]


@torch.no_grad()
def predicted_measurements(predictions, fold_models):
    """(clips, 8) measurements predicted by the fold model that never saw the clip."""
    rows = []
    for clip in predictions.itertuples():
        model = fold_models.get(clip.cv_fold)
        spectrogram = clip_spectrogram(model, clip, fold_models.device)
        values = model(spectrogram[None])['measurements'][0].float().cpu().numpy()
        values[6:] = 1 / (1 + np.exp(-values[6:]))  # the presence outputs are logits
        rows.append(values)
    return pd.DataFrame(rows, columns=MEASUREMENTS, index=predictions.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run')
    parser.add_argument('--measured', default='data/clips_audited_measured.csv')
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictions = load_run_predictions(arguments.run)
    predicted = predicted_measurements(predictions, FoldModels(arguments.run, device))
    measured = read_table(arguments.measured).set_index('name')
    measured = measured.loc[predictions.name, [f'm_{name}' for name in MEASUREMENTS]]
    measured = measured.set_axis(MEASUREMENTS, axis=1).set_index(predictions.index)

    result = {'n_clips': len(predictions), 'accuracy': float(predictions.correct.mean())}
    result['median_abs_error'] = {
        name: float((predicted[name] - measured[name]).abs().median()) for name in CONTINUOUS
    }
    result['presence_accuracy'] = {
        name: float(
            ((predicted[name] > 0.5) == (measured[name] > 0.5))[measured[name].notna()].mean()
        )
        for name in PRESENCE
    }
    result['class_medians'] = {
        label: {
            name: [
                float(measured[name][predictions.label == label].median()),
                float(predicted[name][predictions.label == label].median()),
            ]
            for name in MEASUREMENTS
        }
        for label in CLASSES
    }
    print(json.dumps(result, indent=1))
    with open(f'{WS}/results/measurement_bottleneck_{arguments.run}.json', 'w') as file:
        json.dump(result, file, indent=1)


if __name__ == '__main__':
    main()
