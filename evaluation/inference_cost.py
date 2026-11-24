"""What does one 4 s window cost? Parameters, multiply-adds and single-thread CPU time per model.

    python -m evaluation.inference_cost <run> [<run> ...]
    python -m evaluation.inference_cost --untrained <experiment> [<experiment> ...]

With `--untrained` the model is built by the experiment script's `make_model()` instead of loaded
from a checkpoint: size and speed do not depend on training, so they can be known beforehand.

For every run (its fold-0 checkpoint): the number of parameters, the multiply-accumulate
operations of the network for one window, and the time of one window on ONE CPU thread, split into
the spectrogram front-end and the network. One thread is the honest unit for a phone and for a
shared server; times are the fastest of 30 repetitions, because the machine may be busy.

The last column is the cost of scanning a recording the way the service does (a window every
0.4 s, i.e. 150 windows per minute of audio): seconds of one CPU thread per minute of audio.
"""

import importlib
import os
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from config import WS, effective_run_name
from training.checkpoint import load_model

WINDOWS_PER_MINUTE = 150  # service scan: one 4 s window every 0.4 s


def fastest(function, repetitions=30):
    """Shortest wall time of `function()` over several repetitions, in milliseconds."""
    best = float('inf')
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best * 1000


def cost_of(run):
    """Cost of a trained run (its fold-0 checkpoint)."""
    return cost_of_model(load_model(f'{WS}/runs/{effective_run_name(run)}/fold0/model.pt'))


def untrained_model(experiment):
    """The model an experiment script would train, freshly initialised."""
    os.environ['STRNADI_DRY_RUN'] = '1'  # older scripts train on import unless this is set
    module = importlib.import_module(f'experiments.{experiment}')
    return module.make_model().eval()


@torch.no_grad()
def cost_of_model(model):
    waveform = torch.randn(1, 128000) * 0.1
    parameters = sum(value.numel() for value in model.parameters())

    # Models that compute their own input from the audio are timed as a whole.
    takes_waveform = getattr(model, 'wave_input', False) or getattr(model, 'frontend', None) is None
    if takes_waveform:
        network_input, frontend_ms = waveform, 0.0
    else:
        network_input = model.frontend(waveform)
        frontend_ms = fastest(lambda: model.frontend(waveform))

    try:
        with FlopCounterMode(display=False) as counter:
            model(network_input)
        gmacs = counter.get_total_flops() / 1e9
    except Exception:  # an operator the counter does not know
        gmacs = float('nan')
    return {
        'parameters_m': parameters / 1e6,
        'network_gmacs': gmacs,
        'frontend_ms': frontend_ms,
        'network_ms': fastest(lambda: model(network_input)),
    }


def main():
    torch.set_num_threads(1)
    print(
        f'{"run":38s}{"params (M)":>11s}{"network GMACs":>15s}{"front-end ms":>14s}'
        f'{"network ms":>12s}{"window ms":>11s}{"s per min of audio":>20s}'
    )
    untrained = '--untrained' in sys.argv
    for run in [argument for argument in sys.argv[1:] if not argument.startswith('--')]:
        try:
            cost = cost_of_model(untrained_model(run)) if untrained else cost_of(run)
        except Exception as error:
            print(f'{run:38s} could not be measured: {type(error).__name__}: {str(error)[:60]}')
            continue
        window_ms = cost['frontend_ms'] + cost['network_ms']
        print(
            f'{run:38s}{cost["parameters_m"]:11.2f}{cost["network_gmacs"]:15.2f}'
            f'{cost["frontend_ms"]:14.1f}{cost["network_ms"]:12.1f}{window_ms:11.1f}'
            f'{window_ms * WINDOWS_PER_MINUTE / 1000:20.1f}'
        )


if __name__ == '__main__':
    main()
