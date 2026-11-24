"""A small detector in front of the classifier: how many songs survive, how much work is saved?

    python -m evaluation.detector_gate <detector run> [<detector run> ...]
                                       [--classifier runs/strnadnet_field_ready/final/model.pt]

The service scans a recording with the classifier (a 4 s window every 0.4 s). With a gate, a much
smaller detector scans instead, and the classifier only looks at the windows the detector flags
(plus one window on either side, so that the vote curve of a song stays whole). Songs are then
found and labelled exactly as without the gate (`serving/scan.py`).

Measured on the whole recordings of the held-out Vizina males (663 annotated songs), for several
detector thresholds on `1 - p(None)`:

    songs found, right label      as in evaluation/whole_recording.py
    windows to the classifier     share of all windows the classifier still has to process
    CPU s per min of audio        detector on every window + classifier on the flagged ones, one
                                  thread (the classifier alone is the first row)
"""

import argparse

import numpy as np
import torch

from config import CLASSES, NONE_IDX, WS, effective_run_name, workspace_path
from evaluation.inference_cost import WINDOWS_PER_MINUTE, cost_of_model
from evaluation.whole_recording import scan_recording
from serving.scan import detect_songs
from training.checkpoint import load_model, run_checkpoints
from training.data import read_table

HOP_SECONDS = 0.4
THRESHOLDS = (0.5, 0.2, 0.05)


def window_ms(model):
    """Single-thread time of one window on the CPU."""
    cost = cost_of_model(model.cpu())
    return cost['frontend_ms'] + cost['network_ms']


def score(songs, scans, flagged_by_recording):
    """Songs found / labelled right / unmatched per minute, with the classifier gated."""
    n_songs = found = right = unmatched = 0
    n_windows = n_flagged = 0
    for npy, recording_songs in songs.groupby('npy'):
        starts, probabilities, attention_peak = scans[npy]
        flagged = flagged_by_recording[npy]
        gated = probabilities.copy()
        gated[~flagged] = np.eye(len(CLASSES))[NONE_IDX]  # unseen windows count as "no song"
        detections = detect_songs(starts, gated, attention_peak, HOP_SECONDS, NONE_IDX)
        n_windows += len(starts)
        n_flagged += int(flagged.sum())

        used = np.zeros(len(detections), bool)
        for song in recording_songs.itertuples():
            n_songs += 1
            hits = [
                index
                for index, (time, _, _) in enumerate(detections)
                if song.t0 + 0.3 <= time <= song.t1 + 0.6 and not used[index]
            ]
            if hits:
                used[hits[0]] = True
                found += 1
                right += CLASSES[detections[hits[0]][1]] == song.label
        unmatched += int((~used).sum())
    minutes = n_windows * HOP_SECONDS / 60
    return found / n_songs, right / max(found, 1), unmatched / minutes, n_flagged / n_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('detectors', nargs='+', help='detector runs')
    parser.add_argument('--classifier', default='runs/strnadnet_field_ready/final/model.pt')
    options = parser.parse_args()
    arguments = options.detectors
    classifier_path = workspace_path(options.classifier)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    songs = read_table('data/vizina_eval.csv')
    songs = songs[songs.split == 'test']
    recordings = sorted(songs.npy.unique())

    classifier = load_model(classifier_path)
    classifier_ms = window_ms(classifier)
    classifier.to(device)
    scans = {npy: scan_recording([classifier], npy, HOP_SECONDS, device) for npy in recordings}

    header = f'{"detector":28s}{"params":>9s}{"threshold":>10s}{"found":>8s}{"right":>8s}'
    print(header + f'{"unmatched/min":>15s}{"to classifier":>15s}{"CPU s/min":>11s}')
    everything = {npy: np.ones(len(scans[npy][0]), bool) for npy in recordings}
    found, right, unmatched, _ = score(songs, scans, everything)
    print(
        f'{"none (classifier alone)":28s}{"":>9s}{"":>10s}{found:8.3f}{right:8.3f}{unmatched:15.2f}'
        f'{1.0:15.2f}{classifier_ms * WINDOWS_PER_MINUTE / 1000:11.2f}'
    )

    for run in arguments:
        models = [
            load_model(path) for path in run_checkpoints(f'{WS}/runs/{effective_run_name(run)}')
        ]
        detector_ms = window_ms(models[0])
        parameters = sum(value.numel() for value in models[0].parameters())
        models = [model.to(device) for model in models]
        p_song = {
            npy: 1 - scan_recording(models, npy, HOP_SECONDS, device)[1][:, NONE_IDX]
            for npy in recordings
        }
        for threshold in THRESHOLDS:
            flagged = {}
            for npy in recordings:
                fires = p_song[npy] >= threshold
                neighbours = np.convolve(fires, np.ones(3), mode='same') > 0  # one window each side
                flagged[npy] = neighbours
            found, right, unmatched, share = score(songs, scans, flagged)
            cpu_seconds = (detector_ms + classifier_ms * share) * WINDOWS_PER_MINUTE / 1000
            print(
                f'{run[:27]:28s}{parameters:9,d}{threshold:10.2f}{found:8.3f}{right:8.3f}'
                f'{unmatched:15.2f}{share:15.2f}{cpu_seconds:11.2f}'
            )


if __name__ == '__main__':
    main()
