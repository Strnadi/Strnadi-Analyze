"""End-to-end check of the service code on whole recordings with annotated songs.

    python -m tests.check_service_on_recordings <model.onnx> [n_recordings, default 25]

Runs `serving.pipeline.SongClassifier.classify`, exactly what `/classify` runs, on recordings of
the held-out Vizina males and compares the returned segments with the ornithologists' annotation:
how many annotated songs get a segment, and how many of those segments carry the right label.
(The same test with the PyTorch model on all recordings: `evaluation/whole_recording.py`.)
"""

import sys

import numpy as np

from config import CLASSES
from serving.pipeline import SongClassifier
from training.data import read_table


def main():
    model_path = sys.argv[1]
    n_recordings = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    classifier = SongClassifier(model_path, CLASSES, threads=4)
    print('scanning with attention voting' if classifier.has_attention else 'segmenter fallback')

    songs = read_table('data/vizina_eval.csv')
    songs = songs[songs.split == 'test']
    recordings = sorted(songs.npy.unique())[:n_recordings]

    n_songs, found, right, extra, minutes = 0, 0, 0, 0, 0.0
    for npy in recordings:
        audio = np.load(npy).astype(np.float32) / 32768.0
        minutes += len(audio) / 32000 / 60
        segments = classifier.classify(audio)['segments']
        used = np.zeros(len(segments), bool)
        for song in songs[songs.npy == npy].itertuples():
            n_songs += 1
            for index, segment in enumerate(segments):
                start, end = segment['interval']
                overlap = min(end, song.t1) - max(start, song.t0)
                if overlap >= 0.5 * (song.t1 - song.t0) and not used[index]:
                    used[index] = True
                    found += 1
                    # "Unknown" is a dialect the service was not sure enough to name
                    right += segment['label'] == song.label
                    break
        extra += int((~used).sum())

    print(f'{len(recordings)} recordings, {minutes:.0f} min, {n_songs} annotated songs')
    print(f'annotated songs with a segment: {found / n_songs:.3f}')
    print(f'of those, right label: {right / max(found, 1):.3f}')
    print(f'segments matching no annotated song: {extra / minutes:.2f} per minute')


if __name__ == '__main__':
    main()
