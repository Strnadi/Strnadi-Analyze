"""Expert element timing for the focal recordings, as a training table.

    python -m processing.build_element_spans [--out data/vizina_element_spans.csv]

For every annotated song of the TRAINING males at the Vizina site the ornithologists marked the
first phrase and the terminal components. The parsed annotation (`data/vizina_eval.csv`) keeps the
song's start and end, the start of the first terminal component, and which components exist. This
script turns that into one row per song with the columns the trainer reads for the anchored
element-sequence loss (`training/data.py::token_frame_mask`):

    phrase_t0, phrase_t1       where the first phrase is (seconds in the file)
    terminal_t0, terminal_t1   where the terminal elements are (empty span when there are none)
    n_terminal                 0 = phrase only, 1 = one B whistle, 2 = B and the lower whistle

`label` is BhBl for two terminal elements and Unfinished otherwise, as in the clip labels. The
window is 4.4 s around the song (0.6 s before its start), so that training can shift it.
Songs of the held-out males are not included; nothing existing is modified.
"""

import argparse

import numpy as np
import pandas as pd

from config import SR, workspace_path
from training.data import read_table

WINDOW_SECONDS = 4.4
BEFORE_SONG = 0.6


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='data/vizina_element_spans.csv')
    arguments = parser.parse_args()

    songs = read_table('data/vizina_eval.csv')
    songs = songs[(songs.split == 'train') & (songs.has_first == True)]  # noqa: E712

    rows = []
    for song in songs.itertuples():
        file_seconds = len(np.load(song.npy, mmap_mode='r')) / SR
        window_start = song.t0 - BEFORE_SONG
        if window_start < 0 or window_start + WINDOW_SECONDS > file_seconds:
            continue  # the loader would shift the window and the spans would no longer fit
        n_terminal = int(bool(song.has_d1)) + int(bool(song.has_d2))
        has_terminal = n_terminal > 0 and np.isfinite(song.term_t0)
        if n_terminal > 0 and not has_terminal:
            continue
        phrase_end = song.term_t0 if has_terminal else song.t1
        rows.append(
            {
                'npy': song.npy,
                't0': round(window_start, 3),
                't1': round(window_start + WINDOW_SECONDS, 3),
                'label': 'BhBl' if n_terminal == 2 else 'Unfinished',
                'fold': -1,
                'aug': False,
                'name': f'vizspan:{song.song_code}',
                'w': 1.0,
                'sw': 1.0,
                'source': 'vizina_ann',
                'rec': f'vind:{song.indiv}',
                'phrase_t0': round(song.t0, 3),
                'phrase_t1': round(phrase_end, 3),
                'terminal_t0': round(song.term_t0 if has_terminal else song.t1, 3),
                'terminal_t1': round(song.t1, 3),
                'n_terminal': n_terminal,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(workspace_path(arguments.out), index=False)
    print(f'{len(table)} songs written to {arguments.out}')
    print(table.groupby(['label', 'n_terminal']).size().to_string())


if __name__ == '__main__':
    main()
