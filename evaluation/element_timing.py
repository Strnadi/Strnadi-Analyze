"""Do the decoded element tokens sit where the experts marked the elements?

    python -m evaluation.element_timing <run> [<run> ...]

The only expert element annotation on this machine: the focal recordings of ring-coded males, where
ornithologists marked, for every song, the first phrase and the terminal ("dialect") components.
The parsed table keeps, per song, the song's start and end, the start of the first terminal
component, and which components exist. (The component boxes themselves, with their frequency
bounds and the end of each component, are in the raw selection tables, which are not here.)

For the 663 songs of the 15 held-out males this script decodes the element sequence of a model
trained with the CTC loss (`losses/element_ctc.py`) and compares it with the annotation:

    structure      does the number of decoded terminal elements equal the number annotated
                   (0, 1 or 2)?
    phrase         is a `phrase` token decoded, and does it fall inside the annotated phrase?
    terminal onset how far is the first decoded terminal token from the annotated start of the
                   first terminal component?
"""

import sys

import numpy as np
import torch

from config import SR, WS, effective_run_name
from losses.element_ctc import decode_tokens
from training.checkpoint import load_model, run_checkpoints
from training.data import load_seg, read_table
from training.inference import peak_normalize

FRAME_SECONDS = 0.04  # the token head has one frame per 40 ms


def is_true(value):
    """A table flag as a bool, whether the table was read with booleans or with strings."""
    return value is True or str(value) == 'True'


@torch.no_grad()
def element_timing(run, device):
    models = [
        load_model(path).to(device)
        for path in run_checkpoints(f'{WS}/runs/{effective_run_name(run)}')
    ]
    songs = read_table('data/vizina_eval.csv')
    songs = songs[songs.split == 'test']

    structure_right, phrase_found, phrase_inside, onset_errors = [], [], [], []
    # Every fold model decodes every song on its own: token posteriors are single-frame spikes,
    # and averaging the spikes of several models before decoding would blur them away.
    for song, model in ((song, model) for song in songs.itertuples() for model in models):
        window_start = float(song.win_t0)
        waveform = peak_normalize(
            load_seg(song.npy, window_start, window_start + 4.0, 4 * SR)[None]
        )
        batch = torch.from_numpy(waveform).to(device)
        _, spans = decode_tokens(model(batch)['token_log_probs'])
        spans = spans[0]

        # times are relative to the window; a token's time is the middle of its frames
        tokens = [(name, (first + last + 1) / 2 * FRAME_SECONDS) for name, first, last in spans]
        terminals = [time for name, time in tokens if name != 'phrase']
        phrases = [time for name, time in tokens if name == 'phrase']
        n_annotated = int(is_true(song.has_d1)) + int(is_true(song.has_d2))
        structure_right.append(len(terminals) == n_annotated)

        song_start = song.t0 - window_start
        has_terminal_time = song.term_t0 != '' and np.isfinite(float(song.term_t0))
        terminal_start = float(song.term_t0) - window_start if has_terminal_time else np.nan
        phrase_found.append(bool(phrases))
        if phrases and np.isfinite(terminal_start):
            phrase_inside.append(song_start - 0.1 <= phrases[0] <= terminal_start + 0.1)
        if terminals and np.isfinite(terminal_start):
            onset_errors.append(terminals[0] - terminal_start)

    onset_errors = np.array(onset_errors)
    return {
        'songs': len(songs),
        'models': len(models),
        'structure right': float(np.mean(structure_right)),
        'phrase token decoded': float(np.mean(phrase_found)),
        'phrase token inside the annotated phrase': float(np.mean(phrase_inside)),
        'terminal onset: median offset (s)': float(np.median(onset_errors)),
        'terminal onset: median absolute error (s)': float(np.median(np.abs(onset_errors))),
        # how consistent the timing is, whatever its constant delay
        'terminal onset within 0.3 s of the median offset': float(
            np.mean(np.abs(onset_errors - np.median(onset_errors)) <= 0.3)
        ),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {run: element_timing(run, device) for run in sys.argv[1:]}
    print(f'{"":46s}' + ''.join(f'{run[:26]:>28s}' for run in results))
    for measure in next(iter(results.values())):
        print(f'{measure:46s}' + ''.join(f'{results[run][measure]:28.3f}' for run in results))


if __name__ == '__main__':
    main()
