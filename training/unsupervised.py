"""Label-free training of the legacy unsupervised models on detected songs.

    fit_unsupervised(run_name, make_model, objective='reconstruction')   # autoencoders
    fit_unsupervised(run_name, make_model, objective='contrastive')      # embedding models

Songs come from the song index, training split only (no held-out Vizina males, no citizen-recording
audio: the same corpus rule as `training/pretrain.py`). For the contrastive objective the two views
of a song are made at waveform level (speed, device EQ, real noise) and not only by SpecAugment
inside the model, as the Keras script did.

Writes `$STRNADI_WS/ssl/<run_name>/model.pt` and `log.json`.
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from augment.pipeline import WaveAugment, peak_normalize
from config import WS, dry_run, workspace_path
from layers.spec_augment import SpecAugment
from losses.nt_xent import nt_xent
from model.base import count_parameters
from training.checkpoint import save_model
from training.data import load_seg
from training.settings import TrainSettings

WINDOW_SAMPLES = 128000


def load_song_corpus(sources=('vizina', 'audiomoth_bhbl', 'birdnet_yh')):
    """Detected songs that may be used for label-free training."""
    songs = pd.read_parquet(f'{WS}/data/song_index.parquet')
    is_yellowhammer = (songs.yh >= 0.3) | songs.yh.isna()
    fits_window = (songs.seg_t1 - songs.seg_t0) <= 3.8
    has_lead_in = songs.seg_t0 >= 0.6
    keep = (
        songs.source.isin(sources)
        & is_yellowhammer
        & fits_window
        & has_lead_in
        & (songs.split == 'train')
    )
    return songs[keep].reset_index(drop=True)


class SongSampler(torch.utils.data.Dataset):
    """An endless stream of random songs: one clean view, or two augmented views of the same song."""

    def __init__(self, songs, augment, two_views):
        self.songs = songs
        self.augment = augment
        self.two_views = two_views

    def __len__(self):
        return 10**9

    def __getitem__(self, _):
        song = self.songs.iloc[np.random.randint(len(self.songs))]
        start = song.seg_t0 - np.random.uniform(0.2, 0.6)
        waveform = load_seg(song.npy, start, start + 4, WINDOW_SAMPLES)

        if self.two_views:
            first_view = self.augment(waveform.copy())
            second_view = self.augment(waveform.copy())
            return torch.from_numpy(first_view), torch.from_numpy(second_view)
        return torch.from_numpy(peak_normalize(waveform)), 0


def fit_unsupervised(
    run_name,
    make_model,
    objective,
    steps=6000,
    batch_size=32,
    learning_rate=1e-3,
    noise_bank='data/noise_bank_clean.npy',
    threads=4,
    data_workers=2,
):
    """Train a model without labels.

    Args:
        run_name: output directory name under `$STRNADI_WS/ssl/`.
        make_model: function returning the model (an autoencoder or an embedding model).
        objective: 'reconstruction' (MSE between `recon` and `target` in the model's output) or
            'contrastive' (NT-Xent between the embeddings of two augmented views).
        steps: number of optimiser steps.
        noise_bank: background-noise bank for the augmented views (skipped if the file is missing).
    """
    model = make_model()
    print(f'{type(model).__name__}: {count_parameters(model):,} parameters', flush=True)
    if dry_run():
        print(f'[dry run] {run_name}: model built, objective {objective}')
        return

    torch.set_num_threads(threads)
    songs = load_song_corpus()
    print(f'{len(songs)} songs', flush=True)

    noise = None
    if os.path.exists(workspace_path(noise_bank)):
        noise = list(np.load(workspace_path(noise_bank), allow_pickle=True))
    view_settings = TrainSettings(
        noise_probability=0.8, snr_reference='band', random_eq_probability=0.7
    )
    augment = WaveAugment(view_settings, noise_bank=noise)

    two_views = objective == 'contrastive'
    loader = torch.utils.data.DataLoader(
        SongSampler(songs, augment, two_views), batch_size=batch_size, num_workers=data_workers
    )
    batches = iter(loader)

    spec_augment = SpecAugment(fill=getattr(model, 'spec_fill', 'zero')).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    def lr_factor(step):
        warmup = min(1.0, (step + 1) / 300)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))
        return warmup * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    out_dir = f'{WS}/ssl/{run_name}'
    os.makedirs(out_dir, exist_ok=True)
    log = []
    started = time.time()
    model.train()

    for step in range(steps):
        first, second = next(batches)

        if two_views:
            first_embedding = model(spec_augment(model.frontend(first)))['emb']
            second_embedding = model(spec_augment(model.frontend(second)))['emb']
            loss = nt_xent(first_embedding, second_embedding, temperature=0.1)
        else:
            outputs = model(first)
            loss = F.mse_loss(outputs['recon'], outputs['target'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            log.append({'step': step, 'loss': loss.item(), 't': time.time() - started})
            print(log[-1], flush=True)

        is_last_step = step == steps - 1
        if (step + 1) % 1000 == 0 or is_last_step:
            save_model(f'{out_dir}/model.pt', model.state_dict(), model, epoch=step + 1)
            with open(f'{out_dir}/log.json', 'w') as file:
                json.dump(log, file)
