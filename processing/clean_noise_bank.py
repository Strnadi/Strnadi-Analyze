"""Remove chunks that contain Yellowhammer song from the background-noise bank.

Why: `build_noise_bank.py` keeps AudioMoth audio at least 10 s away from any BirdNET Yellowhammer detection, but
BirdNET finds only ~75-80 % of songs on these recordings, and the recorders stand in Yellowhammer territories.
The fork review scored 400 chunks with a trained model: 36 were predicted to contain a song, and of the 12 most
confident at least 5 are clear songs with terminal elements. Those chunks were mixed into half of all training
clips, including clips labelled "Unfinished" and "None": a hidden complete BhBl song in the background of an
"Unfinished" clip teaches exactly the wrong thing about the one decision the model is weak at.

A chunk is dropped if a trained model gives p(song) = 1 - p(None) above `--p_song` on any 4 s window of it.
A sheet of the dropped chunks is written for visual confirmation (figs/qa/noise_bank_dropped.png).

Usage: python -m processing.clean_noise_bank --model runs/r2_s0 [--bank data/noise_bank_audiomoth.npy]
                                              [--out data/noise_bank_clean.npy] [--p_song 0.3]
"""

import argparse
import os
import numpy as np
import torch
from config import WS, NONE_IDX
from config import workspace_path
from training.inference import TorchEnsemble, peak_normalize

N = 128000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--bank', default='data/noise_bank_audiomoth.npy')
    ap.add_argument('--out', default='data/noise_bank_clean.npy')
    ap.add_argument('--p_song', type=float, default=0.3)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    bank = list(np.load(workspace_path(a.bank), allow_pickle=True))
    model = TorchEnsemble(workspace_path(a.model))
    p_song = np.zeros(len(bank), np.float32)
    for i in range(0, len(bank), 32):
        chunk = bank[i : i + 32]
        starts = [
            s for c in chunk for s in sorted({0, max(0, len(c) - N)})
        ]  # first and last 4 s of each chunk
        owner = [j for j, c in enumerate(chunk) for _ in sorted({0, max(0, len(c) - N)})]
        X = peak_normalize(
            np.stack(
                [
                    np.resize(np.asarray(chunk[j][s : s + N], np.float32), N)
                    for j, s in zip(owner, starts)
                ]
            )
        )
        ps = 1 - model(X)[:, NONE_IDX]
        for j, v in zip(owner, ps):
            p_song[i + j] = max(p_song[i + j], v)
    keep = p_song <= a.p_song
    clean = np.empty(int(keep.sum()), dtype=object)
    clean[:] = [b for b, k in zip(bank, keep) if k]
    np.save(workspace_path(a.out), clean, allow_pickle=True)
    np.save(workspace_path(a.out).replace('.npy', '_p_song.npy'), p_song)
    print(
        f'kept {int(keep.sum())} of {len(bank)} chunks; dropped {int((~keep).sum())} with p(song) > {a.p_song}'
    )
    try:  # visual check of what was removed
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.signal import stft

        worst = np.argsort(-p_song)[:24]
        fig, ax = plt.subplots(4, 6, figsize=(22, 9))
        for k, j in enumerate(worst):
            f, t, Z = stft(np.asarray(bank[j], np.float32), 32000, nperseg=1024, noverlap=896)
            m = (f >= 2000) & (f <= 9500)
            P = 10 * np.log10(np.abs(Z[m]) ** 2 + 1e-12)
            P -= np.median(P, 1, keepdims=True)
            ax[k // 6, k % 6].imshow(
                P,
                origin='lower',
                aspect='auto',
                extent=[0, t[-1], 2, 9.5],
                vmin=0,
                vmax=30,
                cmap='magma',
            )
            ax[k // 6, k % 6].set_title(
                f'#{j} p(song)={p_song[j]:.2f} {"DROPPED" if not keep[j] else "kept"}', fontsize=8
            )
        os.makedirs(f'{WS}/figs/qa', exist_ok=True)
        fig.tight_layout()
        fig.savefig(f'{WS}/figs/qa/noise_bank_dropped.png', dpi=70)
        print(f'look at {WS}/figs/qa/noise_bank_dropped.png before using the bank')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
