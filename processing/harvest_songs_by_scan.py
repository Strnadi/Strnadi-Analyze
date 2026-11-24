"""Find songs in every continuous recording by scanning it with the trained model.

    python -m processing.harvest_songs_by_scan [--model runs/strnadnet_field_ready/final/model.pt]
                                               [--out data/songs_scan_detected.csv] [--hop 0.4]

The weak and machine-labelled tables hold the songs found by the old chain: energy segmenter ->
BirdNET Yellowhammer score >= 0.3 -> window placement. On recordings where every song is annotated
that chain captures 81 % of the songs; the model's own scan with attention voting
(`evaluation/whole_recording.py::detect_songs`) finds 99.5 %. This script runs the scan over the
citizen archive, the focal recordings and the passive-recorder sample and writes one row per
detected song. **Nothing existing is modified**: the result is a new table.

Each row is a 4.4 s window around a detected song (3.4 s before the detected ending, 1.0 s after
it), with the recording's file- or site-level candidate set as its label (e.g.
`BC|Unfinished|None`; empty when the recording has none). Windows that would contain the ending of
a second song are left out. The column `already_harvested` says whether the old chain had found the
same song; `processing/machine_label_songs.py --songs <this table> --only-new` then labels the
songs that are new.

Only recordings of the training split are scanned (never the held-out Vizina males). A song of a
recording that belongs to cross-validation fold k is never used to train fold k, whatever model
detected it: the trainer drops those rows, as for every extra table.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from config import workspace_path
from evaluation.whole_recording import detect_songs, scan_recording
from training.inference import TorchEnsemble

WINDOW_SECONDS = 4.4
BEFORE_ENDING = 3.4  # seconds of the window before the detected ending (a song is up to ~3 s long)


def recordings_to_scan():
    """One row per continuous recording of the training split: npy, source, rec, label, fold."""
    events = pd.read_parquet(workspace_path('data/song_events.parquet'))
    recordings = events.drop_duplicates('npy')
    continuous = recordings.source.isin(['ncs', 'vizina', 'audiomoth_bhbl'])
    recordings = recordings[continuous & (recordings.split == 'train')].copy()
    # BD recordings: the experts re-assign BD mostly to BlBh (as in processing/build_weak_set.py)
    recordings['label'] = recordings.weak.str.replace('BE|BhBl|', 'BE|BlBh|', regex=False)
    return recordings[['npy', 'source', 'rec', 'label', 'fold', 'file_len']], events


def found_by_old_chain(events, npy):
    """Song segments of one recording that the old chain kept: array (n, 2) of start, end."""
    kept = events[
        (events.npy == npy)
        & (events.yh >= 0.3)
        & np.isfinite(events.win_t0)
        & ~events.borrowed.astype(bool)
    ]
    return kept[['seg_t0', 'seg_t1']].values


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='runs/strnadnet_field_ready/final/model.pt')
    parser.add_argument('--out', default='data/songs_scan_detected.csv')
    parser.add_argument('--hop', type=float, default=0.4, help='seconds between window starts')
    parser.add_argument('--limit', type=int, default=0, help='scan only the first N recordings')
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = TorchEnsemble(workspace_path(arguments.model), device=device).models
    recordings, events = recordings_to_scan()
    events_by_file = dict(tuple(events.groupby('npy')))
    if arguments.limit:
        recordings = recordings.sample(arguments.limit, random_state=0)
    print(f'{len(recordings)} recordings, {recordings.file_len.sum() / 3600:.0f} h', flush=True)

    rows = []
    for count, recording in enumerate(recordings.itertuples(), start=1):
        if recording.file_len < WINDOW_SECONDS + 0.1:
            continue
        starts, probabilities, attention_peak = scan_recording(
            models, recording.npy, arguments.hop, device
        )
        if len(starts) == 0:
            continue
        detections = detect_songs(starts, probabilities, attention_peak, arguments.hop)
        endings = np.array([time for time, _ in detections])
        old_segments = found_by_old_chain(
            events_by_file.get(recording.npy, events.iloc[:0]), recording.npy
        )

        for ending in endings:
            window_start = float(
                np.clip(ending - BEFORE_ENDING, 0, recording.file_len - WINDOW_SECONDS)
            )
            window_end = window_start + WINDOW_SECONDS
            other_endings = endings[
                (endings > window_start) & (endings < window_end) & (endings != ending)
            ]
            if len(other_endings):
                continue  # a second song in the window: its label would be ambiguous

            song_span = (ending - 3.0, ending + 0.3)
            overlap = np.minimum(old_segments[:, 1], song_span[1]) - np.maximum(
                old_segments[:, 0], song_span[0]
            )
            rows.append(
                {
                    'npy': recording.npy,
                    't0': round(window_start, 2),
                    't1': round(window_end, 2),
                    'label': recording.label,
                    'fold': recording.fold,
                    'aug': False,
                    'name': f'scan:{recording.rec}@{ending:.2f}',
                    'w': 1.0,
                    'source': recording.source,
                    'rec': recording.rec,
                    'already_harvested': bool((overlap >= 0.5).any()),
                }
            )
        if count % 250 == 0:
            print(f'{count} / {len(recordings)} recordings, {len(rows)} songs', flush=True)

    songs = pd.DataFrame(rows)
    # sampling weight: a recording with many songs should not dominate (as in the weak tables)
    songs['sw'] = 1 / np.sqrt(songs.groupby('rec').rec.transform('size'))
    songs.to_csv(workspace_path(arguments.out), index=False)

    print(f'\n{len(songs)} songs written to {arguments.out}')
    print(pd.crosstab(songs.source, songs.already_harvested).to_string())


if __name__ == '__main__':
    main()
