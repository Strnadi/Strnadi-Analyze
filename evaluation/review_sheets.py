"""Spectrogram sheets for judging clip labels by eye.

    python -m evaluation.review_sheets <run> [<run> ...] [--out figs/label_review] [--agreeing 0]

Averages the out-of-fold predictions of the given runs, lists every clip whose averaged prediction
differs from its label, and draws them ten to a sheet. Each panel shows the clip **inside its source
recording**, with context before and after, because the commonest label error cannot be judged
from the clip alone: "is the ending missing, or did the clip cut it off?".

    dashed cyan lines    the clip as it was labelled (4 s)
    dotted green lines   the window the models are trained on (re-cut around the song)

Next to the sheets it writes `review.csv` (one row per panel, with an empty `verdict` column to fill
in). `docs/LABEL_AUDIT.md` describes the rules used for the audit of 2026-11-24.

`--agreeing N` adds N random clips per class on which models and label agree: the control sample.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import CLASSES, SR, WS, workspace_path
from evaluation.report import load_run_predictions

CONTEXT_BEFORE = 1.5  # seconds of the source recording drawn before the clip
CONTEXT_AFTER = 2.0
PANELS_PER_SHEET = 10


# ------------------------------------------------------------------------------- what to review
def averaged_predictions(runs):
    """Average the out-of-fold class probabilities of several runs, clip by clip."""
    probability_columns = [f'p_{name}' for name in CLASSES]
    tables = [load_run_predictions(run) for run in runs]

    shared = tables[0].index
    for table in tables[1:]:
        shared = shared.intersection(table.index)

    clips = tables[0].loc[shared].copy()
    mean = np.mean([table.loc[shared, probability_columns].values for table in tables], axis=0)
    clips[probability_columns] = mean
    clips['predicted'] = [CLASSES[index] for index in mean.argmax(1)]
    clips['confidence'] = mean.max(1)
    clips['p_label'] = mean[np.arange(len(clips)), clips.label.map(CLASSES.index).values]
    return clips


def locate_in_source(clips):
    """Add `source_npy` and `source_start`: where each labelled 4 s clip sits in its recording.

    Clips of the citizen archive were located in their source recording by the realignment step
    (`src_off`); clips from the BirdNET app have no longer recording and are drawn as they are.
    """
    archive = {
        os.path.basename(path).lower(): path for path in glob.glob(f'{WS}/audio32k/ncs/*.npy')
    }
    source_files, source_starts = [], []
    for clip in clips.itertuples():
        located = clip.src == 'ncs' and str(clip.src_off) not in ('', 'nan')
        if not located:
            source_files.append(None)
            source_starts.append(np.nan)
            continue
        if '/ncs/' in clip.npy:
            source_files.append(clip.npy)
        else:
            recording = clip.name.lower().split('.wav')[0] + '.wav.npy'
            source_files.append(archive.get(recording))
        source_starts.append(float(clip.src_off))

    clips['source_npy'] = source_files
    clips['source_start'] = source_starts
    return clips


# ------------------------------------------------------------------------------------- drawing
def read_audio(npy, start, end):
    """Samples [start, end) seconds of a cached recording; parts outside the file are faint noise."""
    recording = np.load(npy, mmap_mode='r')
    first, last = int(round(start * SR)), int(round(end * SR))
    inside_first, inside_last = max(first, 0), min(last, len(recording))

    audio = np.random.default_rng(0).normal(0, 1e-5, last - first).astype(np.float32)
    if inside_last > inside_first:
        samples = np.asarray(recording[inside_first:inside_last], np.float32) / 32768.0
        audio[inside_first - first : inside_last - first] = samples
    return audio


def draw_spectrogram(ax, audio, title, band_khz=(2.0, 10.0), marks=(), mark_style=None):
    """High-resolution spectrogram (16 ms window, 4 ms hop), each frequency relative to its median."""
    n_fft, hop = 512, 128
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(torch.from_numpy(audio), n_fft, hop, window=window, return_complex=True)
    level_db = 10 * np.log10(spectrum.abs().pow(2).numpy() + 1e-12)

    frequency_khz = np.fft.rfftfreq(n_fft, 1 / SR) / 1000
    in_band = (frequency_khz >= band_khz[0]) & (frequency_khz <= band_khz[1])
    level_db = level_db[in_band]
    level_db = level_db - np.median(level_db, axis=1, keepdims=True)

    ax.imshow(
        level_db,
        origin='lower',
        aspect='auto',
        extent=[0, len(audio) / SR, band_khz[0], band_khz[1]],
        vmin=0,
        vmax=40,
        cmap='magma',
        interpolation='nearest',
    )
    for position in marks:
        ax.axvline(position, **(mark_style or dict(color='c', lw=0.8, ls='--')))
    ax.set_title(title, fontsize=9, loc='left', pad=2)
    ax.set_xticks(np.arange(0, len(audio) / SR + 0.01, 0.5))
    ax.tick_params(labelsize=7)


def draw_clip(ax, clip, title, band_khz=(2.0, 10.0)):
    """One panel: the clip in its source recording, or the bare clip when there is none."""
    if not isinstance(clip.source_npy, str):
        audio = read_audio(clip.npy, clip.t0, clip.t1)
        draw_spectrogram(ax, audio, title + '  [no source recording]', band_khz)
        return

    start = clip.source_start - CONTEXT_BEFORE
    audio = read_audio(clip.source_npy, start, clip.source_start + 4.0 + CONTEXT_AFTER)
    labelled_clip = [CONTEXT_BEFORE, CONTEXT_BEFORE + 4.0]
    draw_spectrogram(ax, audio, title, band_khz, marks=labelled_clip)
    if '/ncs/' in clip.npy:
        for position in (clip.t0 - start, clip.t1 - start):
            ax.axvline(position, color='lime', lw=0.8, ls=':')


def draw_sheets(clips, out_dir, band_khz=(2.0, 10.0)):
    """Write `sheet_000.png`, ... with ten panels each; panel titles carry the row's `review_id`."""
    os.makedirs(out_dir, exist_ok=True)
    for first in range(0, len(clips), PANELS_PER_SHEET):
        figure, axes = plt.subplots(5, 2, figsize=(18, 13), squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, clip in zip(
            axes.ravel(), clips.iloc[first : first + PANELS_PER_SHEET].itertuples()
        ):
            ax.axis('on')
            title = (
                f'#{clip.review_id}  label={clip.label}  models={clip.predicted}'
                f'({clip.confidence:.2f})  p(label)={clip.p_label:.2f}  {clip.name[:30]}'
            )
            draw_clip(ax, clip, title, band_khz)
        figure.tight_layout(pad=0.3)
        figure.savefig(f'{out_dir}/sheet_{first // PANELS_PER_SHEET:03d}.png', dpi=80)
        plt.close(figure)


# ---------------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--out', default='figs/label_review')
    parser.add_argument('--agreeing', type=int, default=0, help='control clips per class')
    parser.add_argument('--skip', help='CSV with a `name` column: clips reviewed before')
    arguments = parser.parse_args()

    clips = locate_in_source(averaged_predictions(arguments.runs))
    if arguments.skip:
        reviewed_before = set(pd.read_csv(arguments.skip, keep_default_na=False).name)
        clips = clips[~clips.name.isin(reviewed_before)]

    disputed = clips[clips.predicted != clips.label].sort_values(['label', 'predicted', 'p_label'])
    parts = [disputed.assign(kind='disputed')]
    if arguments.agreeing:
        agreeing = clips[clips.predicted == clips.label]
        for _, of_class in agreeing.groupby('label'):
            sample = of_class.sample(min(arguments.agreeing, len(of_class)), random_state=0)
            parts.append(sample.assign(kind='control'))

    review = pd.concat(parts).reset_index(drop=True)
    review['review_id'] = np.arange(len(review))
    review['verdict'] = ''

    out_dir = workspace_path(arguments.out)
    draw_sheets(review, out_dir)
    columns = [
        'review_id',
        'kind',
        'name',
        'label',
        'predicted',
        'confidence',
        'p_label',
        'verdict',
    ]
    review[columns].to_csv(f'{out_dir}/review.csv', index=False)
    print(
        f'{len(disputed)} disputed clips, {len(review) - len(disputed)} control clips -> {out_dir}'
    )


if __name__ == '__main__':
    main()
