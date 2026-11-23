"""Bird-identity embedding for diarisation ("which bird sang which song"), trained on Vizina males.

    fit_identity(run_name, make_model, steps=4000)

Songs: Vizina song-index rows whose file names a single ring-coded male, split by male (the held-out
males are never seen in training).
Batches: 12 males x 4 songs. Loss: cosine-margin classification over male identities plus a
supervised-contrastive term on the same embeddings.

Evaluation on the held-out males:
- verification: equal error rate over all song pairs ("same bird or not?");
- synthetic diarisation: take 8 songs each of 1-3 random held-out males, cluster their embeddings
  (agglomerative, cosine distance, threshold tuned on the training males) and report how often the
  number of birds is right and how pure the clusters are.

Audit note: male codes stop carrying over between years in 2024, so "different males" across that
boundary may be the same birds.
"""

import json
import math
import os
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from augment.random_eq import random_eq
from config import WS, dry_run, workspace_path
from losses.supcon import supcon
from training.data import load_seg

WINDOW_SAMPLES = 128000
MALES_PER_BATCH = 12
SONGS_PER_MALE = 4
MARGIN = 0.25
SCALE = 24.0


# ------------------------------------------------------------------------------------------ data
def load_male_songs(min_songs=20):
    """Vizina songs from files naming exactly one male, split into training and held-out males.

    Returns:
        (train_songs, test_songs), each with a `male` column.
    """
    songs = pd.read_parquet(f'{WS}/data/song_index.parquet')
    keep = (
        (songs.source == 'vizina')
        & (songs.yh >= 0.3)
        & ((songs.seg_t1 - songs.seg_t0) <= 3.8)
        & (songs.seg_t0 >= 0.6)
    )
    songs = songs[keep]

    files = pd.read_csv(f'{WS}/data/vizina_files.csv', dtype={'inds': str})
    names_one_male = files.inds.fillna('').str.count(';') == 0
    single_male_files = files[names_one_male].dropna(subset=['inds'])
    male_of_file = dict(zip(single_male_files.npy, single_male_files.inds))

    songs = songs.assign(male=songs.npy.map(male_of_file)).dropna(subset=['male'])
    songs_per_male = songs.groupby('male').size()
    songs = songs[songs.male.isin(songs_per_male[songs_per_male >= min_songs].index)]

    # A quarter of the males, chosen by a hash of their code, are held out.
    males = sorted(songs.male.unique())
    test_males = {male for male in males if zlib.crc32(male.encode()) % 4 == 0}
    train = songs[~songs.male.isin(test_males)].reset_index(drop=True)
    test = songs[songs.male.isin(test_males)].reset_index(drop=True)
    return train, test


def load_window(song, noise_bank, augment):
    """A 4 s window around a song; with `augment`, random position, device EQ and noise."""
    if augment:
        start = song.seg_t0 - np.random.uniform(0.2, 0.6)
    else:
        start = song.seg_t0 - 0.4
    waveform = load_seg(song.npy, start, start + 4, WINDOW_SAMPLES)

    if augment:
        if np.random.rand() < 0.6:
            waveform = random_eq(waveform)
        if noise_bank is not None and np.random.rand() < 0.6:
            noise = noise_bank[np.random.randint(len(noise_bank))].astype(np.float32)
            offset = np.random.randint(0, len(noise) - WINDOW_SAMPLES)
            noise = noise[offset : offset + WINDOW_SAMPLES]
            snr_db = np.random.uniform(0, 20)
            signal_power = (waveform**2).mean() + 1e-9
            noise_power = (noise**2).mean() + 1e-9
            waveform = waveform + noise * np.sqrt(signal_power / noise_power / 10 ** (snr_db / 10))

    peak = max(np.abs(waveform).max(), 1e-6)
    return (waveform / peak * 0.5).astype(np.float32)


@torch.no_grad()
def embed_songs(model, songs, noise_bank=None):
    """L2-normalised embeddings (N, D) of the given songs, without augmentation."""
    model.eval()
    embeddings = []
    for start in range(0, len(songs), 64):
        chunk = songs.iloc[start : start + 64]
        batch = np.stack(
            [load_window(song, noise_bank, augment=False) for song in chunk.itertuples()]
        )
        output = model(torch.from_numpy(batch))['emb']
        embeddings.append(F.normalize(output, dim=1).numpy())
    return np.concatenate(embeddings)


# ------------------------------------------------------------------------------------ evaluation
def verification_eer(embeddings, males):
    """Equal error rate of "same male?" decisions over all pairs of songs."""
    from sklearn.metrics import roc_curve

    similarity = embeddings @ embeddings.T
    upper = np.triu_indices(len(embeddings), 1)
    same_male = (males[:, None] == males[None])[upper]

    false_positive_rate, true_positive_rate, _ = roc_curve(same_male, similarity[upper])
    false_negative_rate = 1 - true_positive_rate
    crossing = np.nanargmin(np.abs(false_positive_rate - false_negative_rate))
    return float((false_positive_rate[crossing] + false_negative_rate[crossing]) / 2)


def simulate_recording(males, rng, songs_per_male=8):
    """Pick 1-3 random males and 8 of each one's songs: indices into the song table, and the males."""
    chosen_males = rng.choice(np.unique(males), rng.integers(1, 4), replace=False)
    song_indices = [rng.choice(np.where(males == male)[0], songs_per_male) for male in chosen_males]
    return np.concatenate(song_indices), chosen_males


def cluster_birds(embeddings, distance_threshold):
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, metric='cosine', linkage='average'
    )
    return clustering.fit_predict(embeddings)


def tune_threshold(embeddings, males, rng, trials=40):
    """The clustering distance threshold that most often finds the right number of birds."""
    best_threshold, best_accuracy = None, -1.0
    for threshold in np.linspace(0.2, 0.9, 15):
        right = []
        for _ in range(trials):
            song_indices, chosen_males = simulate_recording(males, rng)
            clusters = cluster_birds(embeddings[song_indices], threshold)
            right.append(len(set(clusters)) == len(chosen_males))
        if np.mean(right) > best_accuracy:
            best_threshold, best_accuracy = threshold, np.mean(right)
    return float(best_threshold)


def diarisation_scores(embeddings, males, threshold, rng, trials=200):
    """How often the number of birds is right, and the mean cluster purity."""
    right_count, purity = [], []
    for _ in range(trials):
        song_indices, chosen_males = simulate_recording(males, rng)
        clusters = cluster_birds(embeddings[song_indices], threshold)
        right_count.append(len(set(clusters)) == len(chosen_males))

        true_males = males[song_indices]
        majority_total = 0
        for cluster in set(clusters):
            majority_total += pd.Series(true_males[clusters == cluster]).value_counts().iloc[0]
        purity.append(majority_total / len(song_indices))
    return float(np.mean(right_count)), float(np.mean(purity))


# -------------------------------------------------------------------------------------- training
def fit_identity(
    run_name,
    make_model,
    steps=4000,
    learning_rate=1e-3,
    noise_bank='data/noise_bank_clean.npy',
    threads=4,
):
    """Train and evaluate the identity embedding.

    Args:
        run_name: name for `identity/<run_name>.pt` and `results/identity_<run_name>.json`.
        make_model: function taking the number of training males and returning a model with a
            cosine head (its `logits` are scaled cosines, its `emb` the embedding).
        steps: optimiser steps.
    """
    if dry_run():
        model = make_model(10)
        outputs = model(torch.randn(2, WINDOW_SAMPLES) * 0.1)
        print(f'[dry run] {run_name}: logits {tuple(outputs["logits"].shape)}')
        return None

    torch.set_num_threads(threads)
    rng = np.random.default_rng(0)

    train, test = load_male_songs()
    male_index = {male: index for index, male in enumerate(sorted(train.male.unique()))}
    print(
        f'{len(male_index)} training males ({len(train)} songs), '
        f'{test.male.nunique()} held-out males ({len(test)} songs)',
        flush=True,
    )

    noise = None
    if os.path.exists(workspace_path(noise_bank)):
        noise = list(np.load(workspace_path(noise_bank), allow_pickle=True))

    model = make_model(len(male_index))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

    def lr_factor(step):
        warmup = min(1.0, (step + 1) / 300)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))
        return warmup * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    songs_of_male = train.groupby('male').indices
    started = time.time()

    model.train()
    for step in range(steps):
        # A batch of 12 males x 4 songs.
        batch_males = rng.choice(list(songs_of_male), MALES_PER_BATCH, replace=False)
        rows = []
        for male in batch_males:
            too_few = len(songs_of_male[male]) < SONGS_PER_MALE
            rows += list(rng.choice(songs_of_male[male], SONGS_PER_MALE, replace=too_few))

        waveforms = np.stack([load_window(train.iloc[row], noise, augment=True) for row in rows])
        labels = torch.tensor([male_index[train.male.iloc[row]] for row in rows])

        outputs = model(torch.from_numpy(waveforms))
        one_hot = F.one_hot(labels, len(male_index)).float()
        logits = outputs['logits'] - SCALE * MARGIN * one_hot  # CosFace margin on the true male
        loss = F.cross_entropy(logits, labels) + 0.5 * supcon(outputs['emb'], labels, 0.1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(step, round(loss.item(), 3), round(time.time() - started), flush=True)

    # ---- evaluation on held-out males
    test_sample = (
        test.sample(frac=1, random_state=0).groupby('male').head(60).reset_index(drop=True)
    )
    test_embeddings = embed_songs(model, test_sample)
    test_males = test_sample.male.to_numpy(dtype=object)
    eer = verification_eer(test_embeddings, test_males)

    train_sample = (
        train.sample(frac=1, random_state=0).groupby('male').head(30).reset_index(drop=True)
    )
    train_embeddings = embed_songs(model, train_sample)
    threshold = tune_threshold(train_embeddings, train_sample.male.to_numpy(dtype=object), rng)
    count_accuracy, purity = diarisation_scores(test_embeddings, test_males, threshold, rng)

    result = {
        'name': run_name,
        'males_train': len(male_index),
        'males_test': int(test.male.nunique()),
        'verification_EER': eer,
        'diar_nbirds_acc': count_accuracy,
        'diar_purity': purity,
        'threshold': threshold,
    }
    print(json.dumps(result))

    os.makedirs(f'{WS}/identity', exist_ok=True)
    os.makedirs(f'{WS}/results', exist_ok=True)
    with open(f'{WS}/results/identity_{run_name}.json', 'w') as file:
        json.dump(result, file, indent=1)
    torch.save(
        {'state': model.state_dict(), 'n_males': len(male_index)}, f'{WS}/identity/{run_name}.pt'
    )
    return result
