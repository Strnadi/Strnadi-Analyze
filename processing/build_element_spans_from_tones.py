"""Where the song elements are, without anyone marking them: label + spectrogram.

    python -m processing.build_element_spans_from_tones [--out data/element_spans_from_tones.csv]
                                                        [--sheet figs/element_spans_from_tones.png]

A model trained with CTC says which elements a clip contains, but announces each with a single
spike that is not at the element's start and carries no duration (docs/RESULTS.md, element
sequences). Per-frame training targets need spans. This script builds them for the hand-labelled
clips from two things that are already there:

- **which elements** the clip must contain is known from its label (`BC` = phrase, B, C; ...);
- **where they are** is read off the spectrogram. The terminal elements are sustained tones: a
  stretch where one strong narrow peak holds its pitch. Such stretches are found by a fixed rule
  (`tone_runs`) and matched to the expected elements by pitch, direction and order
  (`ELEMENT_RULES`). The X element, a short stack of harmonics, is taken as the high-band activity
  just before B; the phrase is the activity that leads up to the first terminal element.

A clip is kept only if every element its label requires was found; the others stay without spans
(the per-frame loss ignores them). For "Unfinished" the label does not say whether a B follows
the phrase: a sustained tone in the B band right after the phrase is taken as B.

Output: one row per clip with spans, in the format of an extra training table, with the column
`elements`: JSON `[[name, start_s, end_s], ...]` in seconds of the audio file, element names as in
`losses/element_ctc.py` (phrase, B, C, E, L, H, X). With `--direct-whistle-names` the two whistles
of BhBl and BlBh are written `Bh` and `Bl` instead of `B` followed by `L` (lower) or `H` (higher).
Nothing existing is modified.

Checked against the ornithologists' times on the 663 held-out focal songs (`--check`): the start
of the first terminal element and the end of the song.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from config import SR, workspace_path
from layers.mel_filterbank import _hz_to_mel, _mel_to_hz
from layers.registry import build_frontend
from training.data import load_seg, read_table
from training.inference import peak_normalize

FRAME_SECONDS = 0.01  # the default front-end has one frame per 10 ms
N_BINS, FMIN, FMAX = 128, 1500.0, 11000.0
#: Centre frequency of every mel bin of the default front-end.
BIN_HZ = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_BINS + 2))[1:-1]

LEVEL = 0.6  # a frame is "strong" when its loudest bin is this far above the band's median (log)
MIN_TONE_SECONDS = 0.15
MAX_PEAK_JUMP_BINS = 3  # a tone may move this many bins from one frame to the next
B_BAND_HZ = (3700.0, 6700.0)


# ------------------------------------------------------------------------------ the spectrogram
class Spectrograms:
    """The default front-end of the models (128-bin log-mel, per-band median removed)."""

    def __init__(self, device):
        self.device = device
        self.frontend = build_frontend().to(device).eval()

    @torch.no_grad()
    def __call__(self, npy, start):
        """(bins, frames) of the 4 s starting at `start` seconds of a cached recording."""
        waveform = peak_normalize(load_seg(npy, start, start + 4.0, 4 * SR)[None])
        return self.frontend(torch.from_numpy(waveform).to(self.device))[0, 0].cpu().numpy()


# ----------------------------------------------------------------------------------- the tones
def tone_runs(picture, min_seconds=MIN_TONE_SECONDS, level=LEVEL):
    """Stretches where a strong narrow-band tone holds its pitch.

    Returns a list of dicts: `start`, `end` (seconds in the window), `hz` (median pitch), `slope`
    (pitch at the end minus pitch at the start, Hz) and `level` (mean height above the median).
    """
    in_band = (BIN_HZ >= 2300) & (BIN_HZ <= 9800)
    first_bin = int(np.flatnonzero(in_band)[0])
    band = picture[in_band]
    peak = band.argmax(0)
    energy = band.max(0)
    strong = energy > level

    runs, start = [], None
    for frame in range(len(peak) + 1):
        continues = (
            frame < len(peak)
            and strong[frame]
            and (
                start is None or abs(int(peak[frame]) - int(peak[frame - 1])) <= MAX_PEAK_JUMP_BINS
            )
        )
        if continues and start is None:
            start = frame
        elif not continues and start is not None:
            if (frame - start) * FRAME_SECONDS >= min_seconds:
                pitch = BIN_HZ[peak[start:frame] + first_bin]
                third = max(len(pitch) // 3, 1)
                runs.append(
                    {
                        'start': start * FRAME_SECONDS,
                        'end': frame * FRAME_SECONDS,
                        'hz': float(np.median(pitch)),
                        'slope': float(np.median(pitch[-third:]) - np.median(pitch[:third])),
                        'level': float(energy[start:frame].mean()),
                    }
                )
            start = frame if frame < len(peak) and strong[frame] else None
    return runs


def activity(picture, level=LEVEL, close_gap_seconds=0.2):
    """Frames in which anything in the song band is strong, with short gaps closed."""
    in_band = (BIN_HZ >= 2500) & (BIN_HZ <= 9500)
    active = picture[in_band].max(0) > level
    gap = int(close_gap_seconds / FRAME_SECONDS)
    last_active = -10 * gap
    for frame in np.flatnonzero(active):
        if 1 < frame - last_active <= gap:
            active[last_active:frame] = True
        last_active = frame
    return active


def phrase_before(picture, end_seconds, min_seconds=0.3):
    """The stretch of activity that leads up to `end_seconds`: (start, end) or None."""
    active = activity(picture)
    end = int(round(end_seconds / FRAME_SECONDS))
    frame = min(end, len(active)) - 1
    # the phrase may stop a little before the terminal element starts
    while frame > 0 and not active[frame] and end - frame < 25:
        frame -= 1
    phrase_end = frame + 1
    while frame > 0 and active[frame - 1]:
        frame -= 1
    if (phrase_end - frame) * FRAME_SECONDS < min_seconds:
        return None
    return frame * FRAME_SECONDS, phrase_end * FRAME_SECONDS


def longest_activity(picture, min_seconds=0.5):
    """The longest stretch of activity in the window: a phrase with nothing after it."""
    active = activity(picture)
    best, start = None, None
    for frame in range(len(active) + 1):
        if frame < len(active) and active[frame]:
            start = frame if start is None else start
        elif start is not None:
            if best is None or frame - start > best[1] - best[0]:
                best = (start, frame)
            start = None
    if best is None or (best[1] - best[0]) * FRAME_SECONDS < min_seconds:
        return None
    return best[0] * FRAME_SECONDS, best[1] * FRAME_SECONDS


def x_before(picture, b_start, min_seconds=0.06):
    """The X element: the longest strong stretch above 6.8 kHz in the 0.6 s before B."""
    high = picture[BIN_HZ >= 6800].max(0) > LEVEL
    first = max(int((b_start - 0.6) / FRAME_SECONDS), 0)
    last = int(b_start / FRAME_SECONDS)
    best, start = None, None
    for frame in range(first, last + 1):
        if frame < last and high[frame]:
            start = frame if start is None else start
        elif start is not None:
            if best is None or frame - start > best[1] - best[0]:
                best = (start, frame)
            start = None
    if best is None or (best[1] - best[0]) * FRAME_SECONDS < min_seconds:
        return None
    return best[0] * FRAME_SECONDS, best[1] * FRAME_SECONDS


# ------------------------------------------------------------------------- label -> elements
def is_b(run):
    return B_BAND_HZ[0] <= run['hz'] <= B_BAND_HZ[1] and abs(run['slope']) < 700


#: For every dialect: the name of the element after B, and the test a tone has to pass to be it,
#: given the B before it.
ELEMENT_RULES = {
    # C: a falling whistle from ~9 to ~6.5 kHz, 0.2-1.3 s after the start of B
    'BC': (
        'C',
        lambda b, run: (
            b['start'] + 0.2 <= run['start'] <= b['end'] + 1.3
            and run['hz'] >= 6000
            and run['slope'] < -250
        ),
    ),
    # E: a rising (or level) whistle around 7-8 kHz after B
    'BE': (
        'E',
        lambda b, run: (
            b['start'] + 0.2 <= run['start'] <= b['end'] + 1.3
            and run['hz'] >= 6300
            and run['slope'] > -250
        ),
    ),
    # L: a second whistle below B, right after it
    'BhBl': (
        'L',
        lambda b, run: (
            b['end'] - 0.1 <= run['start'] <= b['end'] + 0.4 and run['hz'] < b['hz'] - 250
        ),
    ),
    # H: a second whistle above B, right after it
    'BlBh': (
        'H',
        lambda b, run: (
            b['end'] - 0.1 <= run['start'] <= b['end'] + 0.5
            and b['hz'] + 250 < run['hz'] < 7500
            and abs(run['slope']) < 700
        ),
    ),
}


def element_spans(picture, label):
    """The elements of a labelled window: [(name, start_s, end_s), ...] in window seconds, in the
    order they sound, or None when an element the label requires cannot be found."""
    if label == 'None':
        return []
    runs = tone_runs(picture)
    # the terminal elements end the song: candidates for B are tried from the last one back,
    # so that a long syllable of the phrase in the B band is not taken for it
    b_candidates = [run for run in runs if is_b(run) and run['end'] - run['start'] >= 0.2][::-1]

    def with_phrase(first_start, elements):
        phrase = phrase_before(picture, first_start)
        if phrase is None:
            return None
        return [('phrase', *phrase)] + elements

    if label in ELEMENT_RULES:
        second_name, is_second = ELEMENT_RULES[label]
        for b in b_candidates:
            seconds = [run for run in runs if run is not b and is_second(b, run)]
            if seconds:
                second = max(seconds, key=lambda run: run['level'])
                b_end = min(b['end'], second['start'])
                found = with_phrase(
                    b['start'],
                    [('B', b['start'], b_end), (second_name, second['start'], second['end'])],
                )
                if found:
                    return found
        return None

    if label == 'XB':
        for b in b_candidates:
            x = x_before(picture, b['start'])
            if x:
                found = with_phrase(x[0], [('X', *x), ('B', b['start'], b['end'])])
                if found:
                    return found
        return None

    # Unfinished: a phrase, and a B when a sustained tone in its band follows the phrase directly
    for b in b_candidates:
        phrase = phrase_before(picture, b['start'])
        if phrase is not None and b['end'] - b['start'] >= 0.25:
            return [('phrase', *phrase), ('B', b['start'], b['end'])]
    phrase = longest_activity(picture)
    return [('phrase', *phrase)] if phrase else None


# ------------------------------------------------------------------------------------ the table
def labelled_window_start(clip):
    return float(clip.t0) + max((float(clip.t1) - float(clip.t0)) - 4.0, 0) / 2


#: The two whistles of BhBl / BlBh under their direct names, instead of `B` then `L` / `H`.
DIRECT_WHISTLE_NAMES = {'BhBl': {'B': 'Bh', 'L': 'Bl'}, 'BlBh': {'B': 'Bl', 'H': 'Bh'}}


def build_table(spectrograms, direct_whistle_names=False):
    """Spans for every hand-labelled clip: the rows with spans, and the share found per class."""
    clips = read_table('data/clips_audited.csv')
    rows = []
    found = {name: [0, 0] for name in sorted(clips.label.unique())}
    for clip in clips.itertuples():
        start = labelled_window_start(clip)
        spans = element_spans(spectrograms(clip.npy, start), clip.label)
        found[clip.label][1] += 1
        if spans is None:
            continue
        found[clip.label][0] += 1
        rename = DIRECT_WHISTLE_NAMES.get(clip.label, {}) if direct_whistle_names else {}
        elements = [
            [rename.get(name, name), round(start + first, 3), round(start + last, 3)]
            for name, first, last in spans
        ]
        rows.append(
            {
                'npy': clip.npy,
                't0': clip.t0,
                't1': clip.t1,
                'label': clip.label,
                'fold': clip.fold,
                'aug': False,
                'name': f'spans:{clip.name}',
                'w': 1.0,
                'sw': 1.0,
                'source': 'element_spans_from_tones',
                'rec': clip.grp,
                'elements': json.dumps(elements),
            }
        )
    return pd.DataFrame(rows), found


# ---------------------------------------------------------------------------- check and sheet
def check_against_experts(spectrograms):
    """Compare the spans with the ornithologists' times on the held-out focal songs."""
    songs = read_table('data/vizina_eval.csv')
    songs = songs[songs.split == 'test']
    onset_errors, end_errors, n_found, n_songs = [], [], 0, 0
    count_right = []
    for song in songs.itertuples():
        n_annotated = int(bool(song.has_d1)) + int(bool(song.has_d2))
        window_start = float(song.win_t0)
        spans = element_spans(spectrograms(song.npy, window_start), song.label)
        n_songs += 1
        if spans is None:
            continue
        n_found += 1
        terminals = [span for span in spans if span[0] != 'phrase']
        count_right.append(len(terminals) == n_annotated)
        if terminals and np.isfinite(float(song.term_t0)):
            onset_errors.append(terminals[0][1] - (float(song.term_t0) - window_start))
            end_errors.append(terminals[-1][2] - (float(song.t1) - window_start))
    onset_errors, end_errors = np.array(onset_errors), np.array(end_errors)
    print(f'held-out focal songs: spans found for {n_found} of {n_songs}')
    print(f'  number of terminal elements as annotated: {np.mean(count_right):.2f}')
    for name, errors in [
        ('start of the first terminal element', onset_errors),
        ('end of the song', end_errors),
    ]:
        print(
            f'  {name}: median error {np.median(errors):+.3f} s, median |error| '
            f'{np.median(np.abs(errors)):.3f} s, within 0.1 s {np.mean(np.abs(errors) <= 0.1):.2f}, '
            f'within 0.2 s {np.mean(np.abs(errors) <= 0.2):.2f}  (n = {len(errors)})'
        )


def draw_sheet(table, spectrograms, path, per_class=6):
    """Spectrograms of random clips of every class with their spans as coloured bars on top."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from evaluation.element_sequences import TOKEN_COLOURS

    chosen = []
    for _, of_class in table.groupby('label'):
        chosen += list(of_class.sample(min(per_class, len(of_class)), random_state=0).itertuples())
    n_columns = 4
    n_rows = int(np.ceil(len(chosen) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(20, 2.6 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, row in zip(axes.ravel(), chosen):
        ax.axis('on')
        start = labelled_window_start(row)
        picture = spectrograms(row.npy, start)
        ax.imshow(
            picture,
            origin='lower',
            aspect='auto',
            extent=[0, 4.0, 0, N_BINS],
            cmap='magma',
            vmin=0,
            vmax=2,
        )
        ax.set_ylim(0, N_BINS * 1.15)
        for name, first, last in json.loads(row.elements):
            colour = TOKEN_COLOURS[{'Bh': 'H', 'Bl': 'L'}.get(name, name)]
            ax.axvspan(first - start, last - start, ymin=1 / 1.15, ymax=1.0, color=colour)
            ax.text(
                (first + last) / 2 - start,
                N_BINS * 1.075,
                'F' if name == 'phrase' else name,
                ha='center',
                va='center',
                fontsize=9,
            )
        ax.set_yticks([])
        ax.set_title(row.label, loc='left', fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=70)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='data/element_spans_from_tones.csv')
    parser.add_argument('--sheet', default='figs/element_spans_from_tones.png')
    parser.add_argument('--check', action='store_true', help='only compare with the expert times')
    parser.add_argument(
        '--direct-whistle-names',
        action='store_true',
        help='name the two whistles of BhBl / BlBh Bh and Bl instead of B and L / H',
    )
    arguments = parser.parse_args()

    spectrograms = Spectrograms('cuda' if torch.cuda.is_available() else 'cpu')
    check_against_experts(spectrograms)
    if arguments.check:
        return

    table, found = build_table(spectrograms, arguments.direct_whistle_names)
    table.to_csv(workspace_path(arguments.out), index=False)
    print(f'\n{len(table)} clips with spans -> {arguments.out}')
    for name, (n_found, n_all) in found.items():
        print(f'  {name:11s} {n_found:5d} of {n_all:5d}  ({n_found / n_all:.0%})')
    draw_sheet(table, spectrograms, workspace_path(arguments.sheet))
    print(f'sheet: {arguments.sheet}')


if __name__ == '__main__':
    main()
