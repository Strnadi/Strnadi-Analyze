"""Bank of unfinished Yellowhammer songs, mixed in as a second bird (`background_song_probability`).

    python -m processing.build_background_song_bank [--n 1200] [--out data/background_song_bank.npy]

Field recordings often hold a neighbour singing behind the bird being recorded. The songs come
from the machine-labelled table (`processing/machine_label_songs.py`): class Unfinished, confidence
at least 0.95, and only from sources outside the cross-validation folds (BirdNET detections, focal
and passive recordings), so the bank shares no audio with any test fold. Unfinished songs are used
because a quieter unfinished song never changes the label of the clip it is added to.
"""

import argparse

import numpy as np

from config import workspace_path
from training.data import load_seg, read_table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--n', type=int, default=1200)
    parser.add_argument('--out', default='data/background_song_bank.npy')
    arguments = parser.parse_args()

    songs = read_table('data/songs_machine_labelled.csv')
    usable = (songs.label == 'Unfinished') & (songs.confidence >= 0.95) & (songs.fold == -1)
    chosen = songs[usable].sample(arguments.n, random_state=0)

    bank = np.empty(len(chosen), dtype=object)
    for index, song in enumerate(chosen.itertuples()):
        start = song.t0 + max((song.t1 - song.t0) - 4.0, 0) / 2
        bank[index] = load_seg(song.npy, start, start + 4.0, 128000).astype(np.float16)
    np.save(workspace_path(arguments.out), bank, allow_pickle=True)
    print(f'{len(bank)} songs written to {arguments.out}:', chosen.source.value_counts().to_dict())


if __name__ == '__main__':
    main()
