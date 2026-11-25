"""The quantities ornithologists measure on a song ending, read off the spectrogram automatically.

    python -m processing.build_measurements_from_tones [--out data/clips_audited_measured.csv]

The concept-bottleneck model (`layers/measurement_head.py`) classifies a clip from eight
measurements of its ending only, so that every decision can be read back ("B ends at 5.1 kHz, the
next element starts 1.9 kHz higher and falls"). It needs clips for which those measurements are
known. Nobody measured ours; this script takes the element spans that
`processing/build_element_spans_from_tones.py` finds from label and spectrogram and measures the
pitch inside them:

    m_f1                    pitch at the end of the first terminal element (kHz)
    m_f2, m_f3              pitch at the start and at the end of the second terminal element
    m_df = f2 - f1, m_bw = f3 - f2
    m_t2                    duration of the second terminal element (s)
    m_has_first, m_has_second   1 / 0: is there a first / a second terminal element

Values that do not exist for a clip (no second element: no f2, f3, df, bw, t2) are left empty and
ignored by the loss, and so are all values of the clips in which the spans were not found.
Output: a copy of `data/clips_audited.csv` with the eight `m_` columns added. Nothing existing is
modified.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from config import workspace_path
from processing.build_element_spans_from_tones import (
    BIN_HZ,
    FRAME_SECONDS,
    Spectrograms,
    element_spans,
    labelled_window_start,
)
from training.data import read_table


def pitch_at_ends(picture, start, end):
    """Pitch (kHz) at the start and at the end of a tone: medians of its first and last third."""
    in_band = (BIN_HZ >= 2300) & (BIN_HZ <= 9800)
    first = int(round(start / FRAME_SECONDS))
    last = max(int(round(end / FRAME_SECONDS)), first + 1)
    pitch = BIN_HZ[in_band][picture[in_band][:, first:last].argmax(0)] / 1000.0
    third = max(len(pitch) // 3, 1)
    return float(np.median(pitch[:third])), float(np.median(pitch[-third:]))


def measurements(picture, spans):
    """The eight measurements of a window from its element spans (missing ones are NaN)."""
    terminals = [span for span in spans if span[0] != 'phrase']
    values = {name: np.nan for name in ('f1', 'f2', 'f3', 'df', 'bw', 't2')}
    values['has_first'] = float(len(terminals) >= 1)
    values['has_second'] = float(len(terminals) >= 2)
    if terminals:
        _, values['f1'] = pitch_at_ends(picture, terminals[0][1], terminals[0][2])
    if len(terminals) >= 2:
        _, start, end = terminals[1]
        values['f2'], values['f3'] = pitch_at_ends(picture, start, end)
        values['df'] = values['f2'] - values['f1']
        values['bw'] = values['f3'] - values['f2']
        values['t2'] = end - start
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='data/clips_audited_measured.csv')
    arguments = parser.parse_args()

    spectrograms = Spectrograms('cuda' if torch.cuda.is_available() else 'cpu')
    clips = read_table('data/clips_audited.csv')
    rows = []
    for clip in clips.itertuples():
        picture = spectrograms(clip.npy, labelled_window_start(clip))
        spans = element_spans(picture, clip.label)
        rows.append(measurements(picture, spans) if spans is not None else {})
    measured = pd.DataFrame(rows, index=clips.index).add_prefix('m_')
    table = pd.concat([clips, measured], axis=1)
    table.to_csv(workspace_path(arguments.out), index=False)
    print(
        f'{int(measured.m_has_first.notna().sum())} of {len(table)} clips measured -> {arguments.out}'
    )
    columns = ['m_f1', 'm_f2', 'm_f3', 'm_df', 'm_bw', 'm_t2']
    print(table.groupby('label')[columns].median().round(2).to_string())


if __name__ == '__main__':
    main()
