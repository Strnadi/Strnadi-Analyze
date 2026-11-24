"""More training data for the song detector, labelled by the classifier's scan.

    python -m processing.build_detector_scan_table [--out data/detector_scan.csv]

The detector tables of `processing/build_detector_tables.py` hold about 6 thousand songs with
labels from people. The classifier's scan finds songs far better than the detection chain those
came from (99.5 % of annotated songs, against 81 %), so its detections are used here as labels for
the much smaller detector. The project owner allowed this for the detector on 2026-11-24 (the
classifiers themselves are still trained without any model-made targets except where stated).

    song       the windows of `data/songs_scan_detected.csv` (processing/harvest_songs_by_scan.py)
    no song    windows of a sample of the same recordings in which the classifier hears nothing:
               p(song) below 0.02 in the window and below 0.2 in every window within 2 s of it

As in the other detector tables, `Unfinished` stands for "song". Rows keep the fold of their
recording; the held-out Vizina males were never scanned. Nothing existing is modified.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from config import NONE_IDX, workspace_path
from evaluation.whole_recording import scan_recording
from processing.harvest_songs_by_scan import recordings_to_scan
from training.data import read_table
from training.inference import TorchEnsemble

HOP_SECONDS = 0.4
WINDOW_SECONDS = 4.4
RECORDINGS_PER_SOURCE = {'ncs': 600, 'vizina': 200, 'audiomoth_bhbl': 30}
MAX_QUIET_WINDOWS_PER_RECORDING = 40


def quiet_windows(starts, p_song, rng):
    """Start times of windows with no song in or near them, at least 4 s apart."""
    reach = int(round(2.0 / HOP_SECONDS))
    loudest_nearby = np.array(
        [p_song[max(0, index - reach) : index + reach + 1].max() for index in range(len(p_song))]
    )
    candidates = np.flatnonzero((p_song < 0.02) & (loudest_nearby < 0.2))
    rng.shuffle(candidates)
    chosen = []
    for index in candidates:
        if all(abs(starts[index] - other) >= 4.0 for other in chosen):
            chosen.append(starts[index])
        if len(chosen) == MAX_QUIET_WINDOWS_PER_RECORDING:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='runs/strnadnet_field_ready/final/model.pt')
    parser.add_argument('--out', default='data/detector_scan.csv')
    arguments = parser.parse_args()

    songs = read_table('data/songs_scan_detected.csv')
    songs = songs.assign(label='Unfinished')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = TorchEnsemble(workspace_path(arguments.model), device=device).models
    recordings, _ = recordings_to_scan()
    rng = np.random.default_rng(0)
    rows = []
    for source, n_recordings in RECORDINGS_PER_SOURCE.items():
        of_source = recordings[
            (recordings.source == source) & (recordings.file_len > WINDOW_SECONDS + 1)
        ]
        sample = of_source.sample(min(n_recordings, len(of_source)), random_state=0)
        for recording in sample.itertuples():
            starts, probabilities, _ = scan_recording(models, recording.npy, HOP_SECONDS, device)
            p_song = 1 - probabilities[:, NONE_IDX]
            for start in quiet_windows(starts, p_song, rng):
                start = min(start, recording.file_len - WINDOW_SECONDS)
                rows.append(
                    {
                        'npy': recording.npy,
                        't0': round(float(start), 2),
                        't1': round(float(start) + WINDOW_SECONDS, 2),
                        'label': 'None',
                        'fold': recording.fold,
                        'aug': False,
                        'name': f'quiet:{recording.rec}@{start:.2f}',
                        'w': 1.0,
                        'source': recording.source,
                        'rec': recording.rec,
                    }
                )
        print(f'{source}: {len(rows)} quiet windows so far', flush=True)

    table = pd.concat([songs, pd.DataFrame(rows)], ignore_index=True)
    # a recording with many windows should not dominate; the two labels are balanced by the sampler
    table['sw'] = 1 / np.sqrt(table.groupby(['rec', 'label']).rec.transform('size'))
    table.to_csv(workspace_path(arguments.out), index=False)
    print(pd.crosstab(table.source, table.label).to_string())


if __name__ == '__main__':
    main()
