"""Cut whole-song windows out of arbitrary recordings (the replacement for the legacy `processing/split.py`).

For every recording: decode to 32 kHz mono -> find song events (tonal-SNR segmenter, or a trained
`song_detector` checkpoint with --detector) -> write one 4.4 s WAV per song, starting 0.6 s before the song onset,
plus a table of the cuts. This is the step continuous recordings (e.g. the Italian Alps data) need before
classification.

What changed from `split.py`, and why:
- windows are placed on SONGS, not on BirdNET's 3 s / 0.5 s grid (20 % of the old clips started mid-song);
- nothing is time-stretched (rubberband stretching to 5 s distorted element durations, which carry dialect);
- nothing is padded with synthetic pink / white noise (the model learned "synthetic padding => XB"); a window
  that would run past the file is shifted back inside it;
- songs longer than 3.8 s are skipped rather than cut.

Usage: python -m processing.cut_songs <in_dir> <out_dir> [--detector runs/<run>/fold0/model.pt]
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from config import SR
from processing.segment import segment_songs

EXT = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
PRE, WIN, MAX_SONG = 0.6, 4.4, 3.8


def find_songs(y, detector=None):
    if detector is not None:
        import torch

        return [
            (t0, t1)
            for t0, t1, _ in detector.detect(torch.from_numpy(y / max(np.abs(y).max(), 1e-6) * 0.5))
        ]
    segs, _ = segment_songs(y, hi=10, lo=4, max_gap=0.3, min_dur=0.4, max_dur=6.0)
    events = []
    for s0, s1, _ in segs:  # merge pieces of one song (gap < 0.6 s, total <= 4 s)
        if events and s0 - events[-1][1] < 0.6 and s1 - events[-1][0] <= 4.0:
            events[-1][1] = s1
        else:
            events.append([s0, s1])
    return [(s0, s1) for s0, s1 in events if s1 - s0 >= 0.6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--detector')
    a = ap.parse_args()
    import librosa, soundfile as sf

    detector = None
    if a.detector:
        from training.checkpoint import load_model

        detector = load_model(a.detector)
    rows = []
    for path in sorted(p for p in Path(a.in_dir).rglob('*') if p.suffix.lower() in EXT):
        y, _ = librosa.load(path, sr=SR, mono=True)
        rel = path.relative_to(a.in_dir).with_suffix('')
        for k, (s0, s1) in enumerate(find_songs(y, detector)):
            if s1 - s0 > MAX_SONG:
                continue
            i0 = int(max(0, min(s0 - PRE, len(y) / SR - WIN)) * SR)
            clip = y[i0 : i0 + int(WIN * SR)]
            out = Path(a.out_dir) / f'{rel}.{k:03d}.wav'
            os.makedirs(out.parent, exist_ok=True)
            sf.write(out, clip, SR, subtype='PCM_16')
            rows.append(dict(source=str(path), clip=str(out), t0=i0 / SR, song_t0=s0, song_t1=s1))
    pd.DataFrame(rows).to_csv(Path(a.out_dir) / 'cuts.csv', index=False)
    print(len(rows), 'songs cut')


if __name__ == '__main__':
    main()
