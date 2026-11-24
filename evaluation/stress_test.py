"""How does accuracy fall as the audio gets worse? A tie-breaker between models that score alike.

    python -m evaluation.stress_test <run> [<run> ...]

On clean held-out clips most recipes are within noise of each other. This test degrades the same
clips in ways none of the models was trained on, and scores every fold model on its own held-out
clips:

    pink noise      added at 15 / 10 / 5 / 0 dB signal-to-noise ratio, measured in the song band
                    (1.5-11 kHz). Training used recorded background noise, never synthetic noise.
    muffled         low-pass at 6 kHz / 5 kHz: a crude stand-in for distance and cheap microphones.
                    The terminal C / E / X elements live at 6.5-9 kHz, so this removes them: the
                    honest answer for such a clip is "Unfinished", and a good model should move
                    there rather than guess a dialect.

Reported per run: accuracy under each condition; for the muffled clips also how many complete songs
are still given a dialect (should fall) and how many of those dialects are wrong (should stay low).
The label of a clip is kept as it is, so "accuracy" under muffling is not a fair target; the
dialect-error rate is the number to read there.
"""

import sys

import numpy as np
import torch

from config import CLASSES, DIALECT_IDX, SR, WS, effective_run_name
from evaluation.report import load_run_predictions
from training.checkpoint import load_model
from training.data import load_seg

NOISE_LEVELS_DB = (15, 10, 5, 0)
LOW_PASS_HZ = (6000, 5000)
BAND_HZ = (1500, 11000)


def clip_waveforms(predictions):
    """The labelled 4 s of every held-out clip: tensor (N, samples)."""
    waveforms = []
    for clip in predictions.itertuples():
        start = clip.t0 + max((clip.t1 - clip.t0) - 4.0, 0) / 2
        waveforms.append(load_seg(clip.npy, start, start + 4.0, 128000))
    return torch.from_numpy(np.stack(waveforms))


def add_pink_noise(waveforms, snr_db, generator):
    """Add 1/f noise so that signal and noise power in the song band have the given ratio."""
    spectrum = torch.fft.rfft(waveforms)
    frequency = torch.fft.rfftfreq(waveforms.shape[1], 1 / SR).to(waveforms.device)
    in_band = (frequency >= BAND_HZ[0]) & (frequency <= BAND_HZ[1])

    noise = torch.randn(waveforms.shape, generator=generator, device=waveforms.device)
    noise_spectrum = torch.fft.rfft(noise) / frequency.clamp_min(20.0).sqrt()

    signal_power = spectrum[:, in_band].abs().pow(2).sum(1, keepdim=True)
    noise_power = noise_spectrum[:, in_band].abs().pow(2).sum(1, keepdim=True)
    gain = (signal_power / noise_power / 10 ** (snr_db / 10)).sqrt()
    return torch.fft.irfft(spectrum + gain * noise_spectrum, n=waveforms.shape[1])


def low_pass(waveforms, cutoff_hz):
    """Remove everything above the cutoff (with a 500 Hz roll-off)."""
    spectrum = torch.fft.rfft(waveforms)
    frequency = torch.fft.rfftfreq(waveforms.shape[1], 1 / SR).to(waveforms.device)
    gain = ((cutoff_hz + 500 - frequency) / 500).clamp(0, 1)
    return torch.fft.irfft(spectrum * gain, n=waveforms.shape[1])


@torch.no_grad()
def predict(models_by_fold, folds, waveforms, device, degrade=None):
    """Predicted class of every clip, each by the model of its own fold.

    `degrade` is applied to every batch on the device before it is classified."""
    predicted = np.zeros(len(waveforms), int)
    for fold, model in models_by_fold.items():
        rows = np.flatnonzero(folds == fold)
        for first in range(0, len(rows), 256):
            batch = waveforms[rows[first : first + 256]].to(device)
            if degrade is not None:
                batch = degrade(batch)
            batch = batch / batch.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) * 0.5
            predicted[rows[first : first + 256]] = model(batch)['logits'].argmax(1).cpu().numpy()
    return predicted


def stress_test(run, device):
    predictions = load_run_predictions(run)
    labels = predictions.label.map(CLASSES.index).values
    folds = predictions.cv_fold.values
    run_dir = f'{WS}/runs/{effective_run_name(run)}'
    models = {
        fold: load_model(f'{run_dir}/fold{fold}/model.pt').to(device) for fold in np.unique(folds)
    }
    waveforms = clip_waveforms(predictions)
    generator = torch.Generator(device=device).manual_seed(0)
    is_complete = np.isin(labels, DIALECT_IDX)

    result = {'clean': float((predict(models, folds, waveforms, device) == labels).mean())}
    for snr_db in NOISE_LEVELS_DB:
        with_noise = lambda batch, snr_db=snr_db: add_pink_noise(batch, snr_db, generator)  # noqa: E731
        predicted = predict(models, folds, waveforms, device, with_noise)
        result[f'noise {snr_db} dB'] = float((predicted == labels).mean())
    for cutoff in LOW_PASS_HZ:
        muffled = lambda batch, cutoff=cutoff: low_pass(batch, cutoff)  # noqa: E731
        predicted = predict(models, folds, waveforms, device, muffled)
        says_dialect = np.isin(predicted, DIALECT_IDX)
        result[f'muffled {cutoff // 1000} kHz: still a dialect'] = float(
            says_dialect[is_complete].mean()
        )
        wrong = says_dialect & is_complete & (predicted != labels)
        result[f'muffled {cutoff // 1000} kHz: wrong dialect'] = float(wrong[is_complete].mean())
    return result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {run: stress_test(run, device) for run in sys.argv[1:]}
    conditions = list(next(iter(results.values())))
    print(f'{"":34s}' + ''.join(f'{run[:21]:>23s}' for run in results))
    for condition in conditions:
        print(f'{condition:34s}' + ''.join(f'{results[run][condition]:23.3f}' for run in results))


if __name__ == '__main__':
    main()
