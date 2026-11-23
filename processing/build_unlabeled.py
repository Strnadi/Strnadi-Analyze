"""Unlabelled song pool for semi-supervised training (FixMatch, pseudo-labels).

    python -m processing.build_unlabeled [out.csv]        # default: data/unlabeled_v4.csv

Needs data/song_events.parquet (python -m processing.song_events).

All detected song events with a clean window placement (processing/song_events.py), tagged with
their fold (NCS: hashed folds), Vizina training males only, at most 20 songs per recording. Songs of
the JCU training recorders are appended when data/jcu_train.csv exists.
"""

import os
import sys

import numpy as np
import pandas as pd

from config import WS

WINDOW = 4.4
SONGS_PER_RECORDING = 20
OUTPUT_COLUMNS = ['npy', 't0', 't1', 'fold', 'rec', 'source', 'name']


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else f'{WS}/data/unlabeled_v4.csv'

    songs = pd.read_parquet(f'{WS}/data/song_events.parquet')
    is_song = (songs.yh >= 0.3) | songs.yh.isna()
    short_enough = (songs.seg_t1 - songs.seg_t0) <= 3.8
    placed = np.isfinite(songs.win_t0) & ~songs.borrowed
    songs = songs[is_song & short_enough & placed]
    songs = songs[(songs.source != 'vizina') | (songs.split == 'train')]
    songs = songs.sample(frac=1, random_state=0).groupby('rec').head(SONGS_PER_RECORDING)
    songs = songs.assign(
        t0=songs.win_t0,
        t1=songs.win_t0 + WINDOW,
        name=songs.rec + '@' + songs.seg_t0.round(2).astype(str),
    )[OUTPUT_COLUMNS]

    jcu_path = f'{WS}/data/jcu_train.csv'
    if os.path.exists(jcu_path):
        jcu = pd.read_csv(jcu_path)
        jcu = jcu[jcu.label != 'None']
        jcu = jcu.sample(frac=1, random_state=0).groupby('rec').head(SONGS_PER_RECORDING)
        songs = pd.concat([songs, jcu[OUTPUT_COLUMNS]], ignore_index=True)

    songs.to_csv(out_path, index=False)
    print(songs.source.value_counts().to_dict(), len(songs))


if __name__ == '__main__':
    main()
