"""Turn the song events into an 'extra' training table: candidate-set labels plus None windows.

    python -m processing.build_weak_set <out.csv> [min_score=0.3] [weak_share=0.5]

Needs data/song_events.parquet (python -m processing.song_events) and data/none_index.parquet.

One 4.4 s window per song event, placed by processing/song_events.py so that no neighbouring song
is cut at its edges and the terminal element stays inside every 4 s training crop (the first
version of this table used a fixed [onset-0.6, onset+3.8] window; research log section 17).
Events whose BirdNET score was probably borrowed from the next song are left out.

Rows of NCS recordings inherit the cross-validation fold of their recording group, so weak windows
of validation recordings are never trained on. Vizina held-out males ('test') are excluded.
The columns song_t0 / song_t1 give the song span in the file (per-frame targets, ending-referenced
noise level).

Tables built this way: weak_v5.csv (the research session's has no song_t0 / song_t1 columns).
"""

import sys
import zlib

import numpy as np
import pandas as pd

from config import SRC, WS

WINDOW = 4.4
N_LABELLED_CLIPS = 3744  # size of the labelled set the sampling weights are scaled against
OUTPUT_COLUMNS = [
    'npy',
    't0',
    't1',
    'label',
    'fold',
    'aug',
    'name',
    'w',
    'sw',
    'source',
    'rec',
    'song_t0',
    'song_t1',
]


def with_window(song_events):
    """Add the window and song-span columns to a selection of events."""
    song_events = song_events.copy()
    song_events['t0'] = song_events.win_t0
    song_events['t1'] = song_events.t0 + WINDOW
    song_events['song_t0'] = song_events.seg_t0
    song_events['song_t1'] = song_events.seg_t1
    return song_events


def expert_dialects():
    """Recording id -> candidate dialects from the expert re-evaluation (Diblikova et al. 2023)."""
    table = pd.read_excel(f'{SRC}/external/diblikova2023/Supplementary_Table_S1.xlsx')
    recording = table.filename.str.extract(r'^(F\d+)')[0]
    candidates = {'BE': 'BE', 'BlBh': 'BlBh', 'transitional': 'BE|BlBh'}
    return {
        name: candidates[dialect]
        for name, dialect in zip(recording, table.new_dialect)
        if isinstance(name, str)
    }


def main():
    out_path = sys.argv[1]
    min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    weak_share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    song_events = pd.read_parquet(f'{WS}/data/song_events.parquet')
    song_events = song_events[np.isfinite(song_events.win_t0) & ~song_events.borrowed]
    usable = (
        (song_events.split == 'train')
        & (song_events.yh >= min_score)
        & ((song_events.seg_t1 - song_events.seg_t0) <= 3.8)
    )

    # ---- songs with a file-level or site-level label
    songs = with_window(song_events[usable & (song_events.weak != '')])
    # A file label "BD" is mostly BlBh after the 2023 re-evaluation (independent review, V8).
    songs['label'] = songs.weak.str.replace('BE|BhBl|', 'BE|BlBh|', regex=False)

    # ---- recordings the experts re-evaluated: their dialect overrides the file label
    new_dialect = expert_dialects()
    recording_id = song_events.rec.str.replace('ncs:', '')
    by_experts = with_window(
        song_events[usable & (song_events.source == 'ncs') & recording_id.isin(new_dialect)]
    )
    by_experts['label'] = (
        by_experts.rec.str.replace('ncs:', '').map(new_dialect) + '|Unfinished|None'
    )
    songs = pd.concat([songs[~songs.rec.isin(by_experts.rec)], by_experts], ignore_index=True)

    # Weak-only NCS recordings get a hashed fold instead of -1 (= always train), so that every
    # weak recording can be evaluated out of fold.
    without_fold = (songs.source == 'ncs') & (songs.fold < 0)
    if without_fold.any():
        hashed = songs.loc[without_fold, 'rec'].map(lambda rec: zlib.crc32(rec.encode()) % 5)
        songs.loc[without_fold, 'fold'] = hashed.astype(int)

    # ---- windows without any Yellowhammer song
    no_song = pd.read_parquet(f'{WS}/data/none_index.parquet')
    no_song = no_song[no_song.split == 'train'].copy()
    no_song['t0'] = no_song.seg_t0
    no_song['t1'] = no_song.seg_t1
    no_song['label'] = 'None'

    table = pd.concat([songs, no_song], ignore_index=True)
    table['aug'] = False
    table['name'] = table.rec + '@' + table.t0.round(2).astype(str)
    table['w'] = np.where(table.label == 'None', 0.5, 1.0)

    # Long AudioMoth / Vizina files would dominate: weight each row by 1 / sqrt(rows of its
    # recording), then scale so that the weak rows make up about `weak_share` of the draws.
    rows_in_recording = table.groupby('rec').npy.transform('size')
    table['sw'] = 1.0 / np.sqrt(rows_in_recording)
    table['sw'] *= (weak_share / (1 - weak_share)) * N_LABELLED_CLIPS / table.sw.sum()

    table[OUTPUT_COLUMNS].to_csv(out_path, index=False)
    print(table.groupby(['source', 'label']).size().unstack(fill_value=0).T)
    print('rows', len(table), 'recs', table.rec.nunique())


if __name__ == '__main__':
    main()
