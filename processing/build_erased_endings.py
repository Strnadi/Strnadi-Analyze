"""Counterfactual "Unfinished" songs: complete songs with their terminal element erased.

    python -m processing.build_erased_endings [--run strnadnet_audited_labels]
                                              [--out data/erased_endings_v2.csv]

Where the terminal element is, nobody has annotated. The trained model knows: its occlusion map
for "is the song complete?" (evaluation/explain_model.py) lights up on the element that completes
the song. This script uses that map to locate the element in every correctly classified complete
song, replaces that stretch of audio with noise shaped like the clip's own background, and keeps
the result as an "Unfinished" example if the model is then sure (p >= 0.8) that the ending is
gone. A look at 15 before / after pairs of the first version showed 12 clean erasures and 3 with
a visible remnant of the element; the stretch was therefore extended by 0.6 s after the marked
part, and the confidence requirement added.

Unlike the treble cut (augment/treble_cut.py), this works for every dialect, including BhBl and
BlBh, whose second whistle lies in the same band as the rest of the song.

Fold hygiene: each clip is explained by the fold model that never trained on its recording, and
the erased clip keeps its source clip's fold and recording, so it is never used to train the fold
it would be tested in. New audio is written under `audio32k/<name of the table>/`; nothing
existing is modified.

`data/erased_endings.csv` (the first version) replaced the whole band with a 10 ms cross-fade and
has a click at the edges of the erased stretch wherever the recording has loud low-frequency
noise; it is kept for the record and not used. Version 2 erases above 1.2 kHz only (`erase`).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from config import CLASSES, DIALECTS, SR, UNFINISHED_IDX, WS, workspace_path
from evaluation.explain_model import FoldModels, occlusion_maps
from evaluation.report import load_run_predictions
from processing.audio_repair import background_noise
from training.data import load_seg
from training.inference import peak_normalize

FRAME_SECONDS = 0.01  # the model's spectrogram has one frame per 10 ms
FADE_SAMPLES = 960  # 30 ms cross-fade at both ends of the erased stretch
MARGIN_BEFORE = 0.1  # seconds erased before the stretch the map marks
# ... and after it: the map's 0.2 s patches often end inside the element, and what follows the
# terminal element is background anyway (a first version left visible remnants of it)
MARGIN_AFTER = 0.6
MAX_ERASED_SECONDS = 1.6
MIN_P_UNFINISHED = 0.8  # the model must be sure the ending is gone


def ending_span(complete_map):
    """Start and end (seconds) of the stretch that carries the evidence for "complete".

    The importance of every time frame is the positive part of the occlusion map summed over
    frequency. The span is the run of frames around the strongest one that keep at least 30 % of
    its importance.
    """
    per_frame = np.clip(complete_map, 0, None).sum(0)
    peak = int(per_frame.argmax())
    important = per_frame >= 0.3 * per_frame[peak]

    first, last = peak, peak
    while first > 0 and important[first - 1]:
        first -= 1
    while last < len(important) - 1 and important[last + 1]:
        last += 1
    start = first * FRAME_SECONDS - MARGIN_BEFORE
    end = (last + 1) * FRAME_SECONDS + MARGIN_AFTER
    if end - start > MAX_ERASED_SECONDS:
        end = start + MAX_ERASED_SECONDS
    return max(start, 0.0), end


def split_at(waveform, cutoff_hz=1200.0, roll_off_hz=300.0):
    """Split a waveform into the part below `cutoff_hz` and the rest; the two add up to it."""
    spectrum = np.fft.rfft(waveform)
    frequency = np.fft.rfftfreq(len(waveform), 1 / SR)
    gain = np.clip((cutoff_hz + roll_off_hz - frequency) / roll_off_hz, 0, 1)
    low = np.fft.irfft(spectrum * gain, n=len(waveform)).astype(np.float32)
    return low, waveform - low


def erase(waveform, start, end):
    """Replace waveform[start:end] (seconds) by noise shaped like the clip's own background.

    Only the band the model listens to (above 1.2 kHz) is replaced. The low band, where wind and
    traffic are several times louder than the song, runs on untouched: cutting it produced a
    click at both ends of the stretch in the first version of the table, a mark the model could
    have learned instead of the missing ending.
    """
    low, high = split_at(waveform)
    first = int(start * SR)
    last = min(int(end * SR), len(waveform))
    remainder = np.concatenate([high[:first], high[last:]])
    _, filler = split_at(background_noise(remainder, len(waveform)))

    # 1 inside the erased stretch, raised-cosine ramps of 30 ms on both sides
    filler_share = np.zeros(len(waveform), np.float32)
    filler_share[first:last] = 1.0
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, FADE_SAMPLES))).astype(np.float32)
    ramp_start = max(first - FADE_SAMPLES, 0)
    filler_share[ramp_start:first] = ramp[FADE_SAMPLES - (first - ramp_start) :]
    ramp_end = min(last + FADE_SAMPLES, len(waveform))
    filler_share[last:ramp_end] = ramp[::-1][: ramp_end - last]
    return low + high * (1 - filler_share) + filler * filler_share


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run', default='strnadnet_audited_labels')
    parser.add_argument('--out', default='data/erased_endings_v2.csv')
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_models = FoldModels(arguments.run, device)
    predictions = load_run_predictions(arguments.run)
    # the map of a clip the model gets wrong, or is unsure about, does not mark the ending
    usable = (
        predictions.label.isin(DIALECTS) & predictions.correct & (predictions.confidence >= 0.8)
    )
    complete = predictions[usable]

    # the audio of every version of the table has its own directory: nothing is overwritten
    audio_dir = f'{WS}/audio32k/{os.path.basename(arguments.out).removesuffix(".csv")}'
    os.makedirs(audio_dir, exist_ok=True)
    rows, n_still_complete = [], 0
    for clip in complete.itertuples():
        model = fold_models.get(clip.cv_fold)
        start_of_window = clip.t0 + max((clip.t1 - clip.t0) - 4.0, 0) / 2
        waveform = load_seg(clip.npy, start_of_window, start_of_window + 4.0, 4 * SR)

        normalised = torch.from_numpy(peak_normalize(waveform[None])).to(device)
        maps, _ = occlusion_maps(model, model.frontend(normalised)[0])
        start, end = ending_span(maps['complete'])
        erased = erase(waveform, start, end)

        # keep it only if the ending is gone as far as the model can tell
        check = torch.from_numpy(peak_normalize(erased[None])).to(device)
        probabilities = torch.softmax(model(check)['logits'], 1)[0].cpu().numpy()
        if probabilities[UNFINISHED_IDX] < MIN_P_UNFINISHED:
            n_still_complete += 1
            continue

        path = f'{audio_dir}/{clip.name}.npy'
        np.save(path, np.round(np.clip(erased, -1, 1) * 32767).astype(np.int16))
        rows.append(
            {
                'npy': path,
                't0': 0.0,
                't1': 4.0,
                'label': 'Unfinished',
                'fold': clip.fold,
                'aug': False,
                'name': f'erased:{clip.name}',
                'w': 1.0,
                'sw': 1.0,
                'source': 'erased_ending',
                'rec': clip.grp,
                'was': clip.label,
                'erased_from': round(start, 2),
                'erased_to': round(end, 2),
                'p_unfinished': float(probabilities[UNFINISHED_IDX]),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(workspace_path(arguments.out), index=False)
    print(
        f'{len(complete)} complete songs: {len(table)} erased endings kept, '
        f'{n_still_complete} dropped (the model still saw a complete song)'
    )
    print(table.was.value_counts().to_string())
    print(f'median erased length {np.median(table.erased_to - table.erased_from):.2f} s')


if __name__ == '__main__':
    assert len(CLASSES) == 7
    main()
