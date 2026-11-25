"""Per-frame element tags: do the runs name the right class, and do they sit on the elements?

    python -m evaluation.element_frames <run> [--sheet figs/element_frames_<run>.png]

For a model built with `element_frames=True` (`model/strnadnet.py`), whose every 40 ms frame is
tagged background / phrase / B / C / E / L / H / X. A run of equal tags is one element with a
start and an end (`losses.element_ctc.decode_frame_tags`); the sequence of runs stands for a class
through the project's table (`ElementGrammar`).

Reported:

- on the held-out clips of every fold: accuracy of the class head, of the class read from the
  runs, how often the two agree and how accurate the answer is when they do; for BhBl / BlBh
  clips, whether the tag of the second whistle (L = lower, H = higher than the B before) agrees
  with the pitch step measured on the spectrogram inside the two runs;
- on the 663 focal songs of held-out males with the ornithologists' times: whether the number of
  terminal elements is right, and how far the start of the first terminal element and the end of
  the last are from the annotated times (the CTC spikes: median distance 0.5 s);
- a sheet of held-out clips with the tags drawn as bars over the spectrogram.
"""

import argparse
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CLASSES, SR, WS, workspace_path
from evaluation.element_sequences import TOKEN_COLOURS
from evaluation.explain_model import FoldModels, clip_spectrogram
from evaluation.report import load_run_predictions
from losses.element_ctc import (
    STRNADI_GRAMMAR,
    canonical_elements,
    decode_frame_tags,
    refine_boundaries,
)
from processing.build_element_spans_from_tones import BIN_HZ
from training.data import load_seg, read_table
from training.inference import peak_normalize

FRAME_SECONDS = 0.04  # the frame series of StrnadNet has one frame per 40 ms
SPECTROGRAM_FRAME_SECONDS = 0.01


def measured_pitch(picture, start, end):
    """Median pitch (Hz) of the loudest tone between `start` and `end` seconds of a window."""
    in_band = (BIN_HZ >= 2300) & (BIN_HZ <= 9800)
    first = int(round(start / SPECTROGRAM_FRAME_SECONDS))
    last = max(int(round(end / SPECTROGRAM_FRAME_SECONDS)), first + 1)
    peak = picture[in_band][:, first:last].argmax(0)
    return float(np.median(BIN_HZ[in_band][peak]))


def second_whistle_check(runs, picture):
    """For a `B` followed by `L` or `H` (or the pair tagged directly `Bh Bl` / `Bl Bh`): the measured pitch step (second minus first, Hz), and
    whether the tag says the same direction. None when the window has no such pair.

    The tag L / H is the model's judgement of "lower / higher than the B before"; with spans the
    step can be measured on the spectrogram, which does not depend on that judgement.
    """
    names = canonical_elements([name for name, _, _ in runs])
    for index in range(len(runs) - 1):
        if names[index] == 'B' and names[index + 1] in ('L', 'H'):
            (_, start, end), (_, next_start, next_end) = runs[index], runs[index + 1]
            first_pitch = measured_pitch(picture, start, end)
            step = measured_pitch(picture, next_start, next_end) - first_pitch
            return step, (step < 0) == (names[index + 1] == 'L')
    return None


def runs_of(model, batch, refine=True):
    """The element runs of one window: [(name, start_s, end_s), ...].

    With `refine` the boundaries are where each element's probability crosses its level
    (`losses.element_ctc.refine_boundaries`); without it, where the most likely tag changes.
    """
    probabilities = torch.softmax(model(batch)['element_logits'][0].float(), 0).T.cpu().numpy()
    runs = decode_frame_tags(probabilities.argmax(1))
    if refine:
        runs = refine_boundaries(probabilities, runs)
    return [(name, first * FRAME_SECONDS, end * FRAME_SECONDS) for name, first, end in runs]


@torch.no_grad()
def on_held_out_clips(predictions, fold_models):
    """Class read from the runs of every held-out clip, next to the class head."""
    sequence_class, all_runs, steps, tag_agrees, measured_class = [], [], [], [], []
    for clip in predictions.itertuples():
        model = fold_models.get(clip.cv_fold)
        spectrogram = clip_spectrogram(model, clip, fold_models.device)
        runs = runs_of(model, spectrogram[None])
        all_runs.append(runs)
        names = canonical_elements([name for name, _, _ in runs])
        from_runs = STRNADI_GRAMMAR.class_of_sequence(names)
        sequence_class.append(from_runs)
        check = second_whistle_check(runs, spectrogram[0].cpu().numpy())
        if check is not None and clip.label in ('BhBl', 'BlBh'):
            steps.append(check[0])
            tag_agrees.append(check[1])
            # the dialect of the two whistles decided by the measured step alone
            measured_class.append(('BhBl' if check[0] < 0 else 'BlBh') == clip.label)
    decoded = predictions.copy()
    decoded['runs'] = all_runs
    decoded['sequence_class'] = sequence_class
    agree = decoded.sequence_class == decoded.predicted
    result = {
        'n_clips': len(decoded),
        'accuracy_head': float(decoded.correct.mean()),
        'accuracy_runs': float((decoded.sequence_class == decoded.label).mean()),
        'agreement': float(agree.mean()),
        'accuracy_when_agreeing': float(decoded.correct[agree].mean()),
        'accuracy_head_when_disagreeing': float(decoded.correct[~agree].mean()),
        # BhBl / BlBh clips in which a B and a second whistle were tagged
        'two_whistles': {
            'n_clips': len(steps),
            'tag_agrees_with_measured_step': float(np.mean(tag_agrees)) if steps else None,
            'dialect_right_by_measured_step': float(np.mean(measured_class)) if steps else None,
            'median_abs_step_hz': float(np.median(np.abs(steps))) if steps else None,
        },
        'recall_runs': {
            name: float((decoded.sequence_class[decoded.label == name] == name).mean())
            for name in CLASSES
        },
    }
    return decoded, result


@torch.no_grad()
def on_expert_songs(fold_models, device):
    """Timing of the runs against the ornithologists' times (held-out focal songs)."""
    songs = read_table('data/vizina_eval.csv')
    songs = songs[songs.split == 'test']
    models = [fold_models.get(fold) for fold in range(5)]
    count_right, onset_errors, end_errors, phrase_overlap = [], [], [], []
    for song, model in ((song, model) for song in songs.itertuples() for model in models):
        window_start = float(song.win_t0)
        waveform = peak_normalize(
            load_seg(song.npy, window_start, window_start + 4.0, 4 * SR)[None]
        )
        runs = runs_of(model, torch.from_numpy(waveform).to(device))
        # the song the annotation is about: the runs from its phrase on
        terminals = [run for run in runs if run[0] != 'phrase']
        n_annotated = int(bool(song.has_d1)) + int(bool(song.has_d2))
        song_start, song_end = float(song.t0) - window_start, float(song.t1) - window_start
        terminals = [run for run in terminals if song_start - 0.2 <= run[1] <= song_end + 0.3]
        count_right.append(len(terminals) == n_annotated)
        if np.isfinite(float(song.term_t0)):
            terminal_start = float(song.term_t0) - window_start
            phrases = [run for run in runs if run[0] == 'phrase']
            if phrases:
                # share of the annotated phrase that is tagged as phrase
                covered = sum(
                    max(0.0, min(end, terminal_start) - max(start, song_start))
                    for _, start, end in phrases
                )
                phrase_overlap.append(covered / max(terminal_start - song_start, 1e-6))
            if terminals:
                onset_errors.append(terminals[0][1] - terminal_start)
                end_errors.append(terminals[-1][2] - song_end)
    onset_errors, end_errors = np.array(onset_errors), np.array(end_errors)
    return {
        'songs': len(songs),
        'number_of_terminal_elements_right': float(np.mean(count_right)),
        'share_of_the_phrase_tagged_phrase': float(np.mean(phrase_overlap)),
        'terminal_onset_median_error_s': float(np.median(onset_errors)),
        'terminal_onset_median_abs_error_s': float(np.median(np.abs(onset_errors))),
        'terminal_onset_within_0.1_s': float(np.mean(np.abs(onset_errors) <= 0.1)),
        'terminal_onset_within_0.2_s': float(np.mean(np.abs(onset_errors) <= 0.2)),
        'song_end_median_abs_error_s': float(np.median(np.abs(end_errors))),
        'song_end_within_0.2_s': float(np.mean(np.abs(end_errors) <= 0.2)),
    }


def draw_sheet(decoded, fold_models, path, per_class=4):
    """Held-out clips of every class: spectrogram, and the tagged elements as bars above it."""
    rows = []
    for name in CLASSES:
        of_class = decoded[decoded.label == name]
        rows += list(of_class.sample(min(per_class, len(of_class)), random_state=1).itertuples())
    n_columns = 4
    n_rows = int(np.ceil(len(rows) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(20, 2.6 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, clip in zip(axes.ravel(), rows):
        ax.axis('on')
        model = fold_models.get(clip.cv_fold)
        picture = clip_spectrogram(model, clip, fold_models.device)[0].cpu().numpy()
        n_bins = picture.shape[0]
        ax.imshow(
            picture,
            origin='lower',
            aspect='auto',
            extent=[0, 4.0, 0, n_bins],
            cmap='magma',
            vmin=0,
            vmax=2,
        )
        ax.set_ylim(0, n_bins * 1.15)
        for name, start, end in clip.runs:
            colour = TOKEN_COLOURS[{'Bh': 'H', 'Bl': 'L'}.get(name, name)]
            ax.axvspan(start, end, ymin=1 / 1.15, ymax=1.0, color=colour)
            ax.text(
                (start + end) / 2,
                n_bins * 1.075,
                'F' if name == 'phrase' else name,
                ha='center',
                va='center',
                fontsize=9,
            )
        ax.set_yticks([])
        ax.set_title(
            f'label {clip.label} | head {clip.predicted} | runs -> {clip.sequence_class}',
            loc='left',
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=70)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run')
    parser.add_argument('--sheet')
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_models = FoldModels(arguments.run, device)
    decoded, result = on_held_out_clips(load_run_predictions(arguments.run), fold_models)
    result['expert_songs'] = on_expert_songs(fold_models, device)
    print(json.dumps(result, indent=1))
    with open(f'{WS}/results/element_frames_{arguments.run}.json', 'w') as file:
        json.dump(result, file, indent=1)

    sheet = workspace_path(arguments.sheet or f'figs/element_frames_{arguments.run}.png')
    draw_sheet(decoded, fold_models, sheet)
    print(f'sheet: {sheet}')


if __name__ == '__main__':
    main()
