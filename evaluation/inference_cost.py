"""What does one 4 s window cost? Parameters, multiply-adds and single-thread CPU time per model.

    python -m evaluation.inference_cost <run> [<run> ...]

For every run (its fold-0 checkpoint): the number of parameters, the multiply-accumulate
operations of the network for one window, and the time of one window on ONE CPU thread, split into
the spectrogram front-end and the network. One thread is the honest unit for a phone and for a
shared server; times are the fastest of 30 repetitions, because the machine may be busy.

The last column is the cost of scanning a recording the way the service does (a window every
0.4 s, i.e. 150 windows per minute of audio): seconds of one CPU thread per minute of audio.
"""

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


@torch.no_grad()
def cost_of(run):
    model = load_model(f'{WS}/runs/{effective_run_name(run)}/fold0/model.pt')
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
    for run in sys.argv[1:]:
        cost = cost_of(run)
        window_ms = cost['frontend_ms'] + cost['network_ms']
        print(
            f'{run:38s}{cost["parameters_m"]:11.2f}{cost["network_gmacs"]:15.2f}'
            f'{cost["frontend_ms"]:14.1f}{cost["network_ms"]:12.1f}{window_ms:11.1f}'
            f'{window_ms * WINDOWS_PER_MINUTE / 1000:20.1f}'
        )


if __name__ == '__main__':
    main()
