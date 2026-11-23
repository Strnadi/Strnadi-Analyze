"""Label-free pretraining of a StrnadNet encoder on all detected Yellowhammer songs.

    pretrain_encoder(run_name, make_model, objectives={'same_song': 1.0, 'masked_frames': 1.0})

Objectives (any weighted mix):

    same_song                 contrastive: two differently augmented views of the same song are
                              positives (noise, device EQ, gain and speed differ per view, so
                              "same recording channel" is not enough to solve it)
    masked_frames             data2vec-style: an EMA teacher sees the song, the student sees it with
                              time spans masked and regresses the teacher's frame features there
    source_prediction         Perch-2.0-style: which recording does this song come from?
    masked_units              HuBERT-style: predict the k-means unit of each masked frame
    masked_reconstruction     masked autoencoder: reconstruct the spectrogram of masked spans
    bottleneck_reconstruction autoencoder: reconstruct the whole spectrogram from the clip embedding

Corpus: the song index, training split only. Held-out Vizina males are never seen, not even without
labels, and citizen-recording (NCS) audio is left out by default so the cross-validation clips stay
unseen. Vizina files naming one male are grouped by that male.

Two findings of the independent review are built in: positives are views of the SAME song (pairing
different songs of one recording pulls complete and unfinished songs together, the very
distinction the classifier has to make), and source prediction is off by default.

Monitoring: every `probe_every` steps a logistic-regression probe on the frozen clip embedding of
the labelled clips (fold 0 held out). Writes `$STRNADI_WS/ssl/<run_name>/encoder.pt` and `log.json`.

Both reviews rate self-supervised pretraining from scratch on CPU as a low-priority bet next to
distilling a strong pretrained teacher. Origin: research session (`ssl_pretrain.py`).
"""

import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from augment.random_eq import random_eq
from augment.speed_perturb import speed_perturb
from config import C2I, WS, dry_run, workspace_path
from losses.supcon import supcon
from model.base import count_parameters
from training.checkpoint import load_encoder_state, save_model
from training.data import load_seg, read_table
from training.metrics import metrics

WINDOW_SAMPLES = 128000
TARGET_BINS, TARGET_FRAMES = (
    128,
    100,
)  # pooled spectrogram that the reconstruction objectives predict

OBJECTIVES = (
    'same_song',
    'masked_frames',
    'source_prediction',
    'masked_units',
    'masked_reconstruction',
    'bottleneck_reconstruction',
)


# ------------------------------------------------------------------------------------------ data
def load_pretraining_songs(sources, min_yellowhammer_score=0.3):
    """Songs for pretraining, with Vizina single-male files regrouped by male."""
    songs = pd.read_parquet(f'{WS}/data/song_index.parquet')
    is_yellowhammer = (songs.yh >= min_yellowhammer_score) | songs.yh.isna()
    fits_window = (songs.seg_t1 - songs.seg_t0) <= 3.8
    songs = songs[songs.source.isin(sources) & is_yellowhammer & fits_window]
    songs = songs[songs.split == 'train'].copy()

    files = pd.read_csv(f'{WS}/data/vizina_files.csv', dtype={'inds': str})
    names_one_male = files.inds.fillna('').str.count(';') == 0
    single_male_files = files[names_one_male].dropna(subset=['inds'])
    male_of_file = songs.npy.map(dict(zip(single_male_files.npy, single_male_files.inds)))

    has_male = male_of_file.notna()
    songs.loc[has_male, 'rec'] = 'vind:' + male_of_file[has_male]
    print(
        f'{len(songs)} songs from {songs.rec.nunique()} sources; '
        f'{int(has_male.sum())} Vizina songs grouped by {male_of_file.nunique()} males',
        flush=True,
    )
    return songs.reset_index(drop=True)


class SongViews(torch.utils.data.Dataset):
    """An endless stream of (view 1, view 2, source id, clean view 2)."""

    def __init__(self, songs, noise_bank, speed_jitter, other_song_probability):
        self.songs = songs
        self.noise_bank = noise_bank
        self.speed_jitter = speed_jitter
        self.other_song_probability = other_song_probability

        self.songs_of_source = songs.groupby('rec').indices
        self.sources = list(self.songs_of_source)
        self.source_id = {source: index for index, source in enumerate(self.sources)}

    def __len__(self):
        return 10**9

    def _crop(self, song_index):
        song = self.songs.iloc[song_index]
        start = song.seg_t0 - np.random.uniform(0.2, 0.6)
        return load_seg(song.npy, start, start + 4, WINDOW_SAMPLES)

    def _augment(self, waveform):
        if np.random.rand() < 0.5:
            waveform = speed_perturb(waveform, self.speed_jitter)
        if np.random.rand() < 0.7:
            waveform = random_eq(waveform)
        if self.noise_bank is not None and np.random.rand() < 0.8:
            noise = self.noise_bank[np.random.randint(len(self.noise_bank))].astype(np.float32)
            offset = np.random.randint(0, len(noise) - WINDOW_SAMPLES)
            noise = noise[offset : offset + WINDOW_SAMPLES]
            snr_db = np.random.uniform(-3, 20)
            signal_power = (waveform**2).mean() + 1e-9
            noise_power = (noise**2).mean() + 1e-9
            waveform = waveform + noise * np.sqrt(signal_power / noise_power / 10 ** (snr_db / 10))
        waveform = waveform * 10 ** (np.random.uniform(-10, 10) / 20)  # random gain
        return _peak_normalize(waveform)

    def __getitem__(self, _):
        source = self.sources[np.random.randint(len(self.sources))]
        candidates = self.songs_of_source[source]

        first = candidates[np.random.randint(len(candidates))]
        second = first
        if len(candidates) > 1 and np.random.rand() < self.other_song_probability:
            second = candidates[np.random.randint(len(candidates))]

        first_wave = self._crop(first)
        second_wave = self._crop(second)
        return (
            torch.from_numpy(self._augment(first_wave)),
            torch.from_numpy(self._augment(second_wave)),
            self.source_id[source],
            torch.from_numpy(_peak_normalize(second_wave)),
        )


def _peak_normalize(waveform):
    peak = np.abs(waveform).max()
    if peak <= 1e-6:
        return waveform.astype(np.float32)
    return (waveform / peak * 0.5).astype(np.float32)


# --------------------------------------------------------------------------------- building blocks
def mask_time_spans(spectrogram, ratio=0.4, span=12):
    """Zero random time spans until `ratio` of the frames are hidden.

    Returns:
        (masked spectrogram, mask (B, T) with True at hidden frames).
    """
    batch_size, _, _, n_frames = spectrogram.shape
    hidden = torch.zeros(batch_size, n_frames, dtype=torch.bool)
    for item in range(batch_size):
        while hidden[item].float().mean() < ratio:
            start = np.random.randint(0, n_frames - span)
            hidden[item, start : start + span] = True
    masked = spectrogram.masked_fill(hidden[:, None, None, :], 0.0)
    return masked, hidden


def mask_at_frame_rate(hidden, n_frames):
    """Bring a (B, T) input-resolution mask to the model's frame rate: (B, n_frames) of 0/1."""
    return F.adaptive_max_pool1d(hidden.float()[:, None], n_frames).squeeze(1)


def masked_mean(per_frame_loss, frame_mask):
    return (per_frame_loss * frame_mask).sum() / frame_mask.sum().clamp_min(1)


def clip_embedding(model, frames):
    pooled, _ = model.pool(frames)
    return model.emb(pooled)


def reconstruction_target(spectrogram):
    """The pooled, standardised spectrogram (B, 128, 100) the reconstruction objectives predict."""
    target = F.adaptive_avg_pool2d(spectrogram.squeeze(1).float(), (TARGET_BINS, TARGET_FRAMES))
    mean = target.mean((1, 2), keepdim=True)
    std = target.std((1, 2), keepdim=True) + 1e-5
    return (target - mean) / std


class BottleneckDecoder(nn.Module):
    """Clip embedding -> the whole pooled spectrogram (B, 128, 100)."""

    def __init__(self, embedding_dim, n_frames=TARGET_FRAMES):
        super().__init__()
        self.n_frames = n_frames
        self.expand = nn.Linear(embedding_dim, 32 * n_frames)
        self.position = nn.Parameter(torch.zeros(1, 32, n_frames))
        self.decode = nn.Sequential(
            nn.Conv1d(32, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, TARGET_BINS, 3, padding=1),
        )

    def forward(self, embedding):
        series = self.expand(embedding).view(-1, 32, self.n_frames) + self.position
        return self.decode(series)


@torch.no_grad()
def linear_probe(model, labelled, held_out_fold=0):
    """How much dialect information the frozen clip embedding carries.

    Fits logistic regression on the labelled clips outside `held_out_fold`.

    Returns:
        (macro-F1, dialect macro-F1) on the held-out fold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    model.eval()
    embeddings = []
    for start in range(0, len(labelled), 64):
        chunk = labelled.iloc[start : start + 64]
        waves = np.stack([load_seg(r.npy, r.t0, r.t1, WINDOW_SAMPLES) for r in chunk.itertuples()])
        waves = waves / np.maximum(np.abs(waves).max(1, keepdims=True), 1e-6) * 0.5
        frames = model.frames(torch.from_numpy(waves.astype(np.float32)))
        embeddings.append(clip_embedding(model, frames).numpy())
    embeddings = np.concatenate(embeddings)
    labels = labelled.label.map(C2I).values

    is_copy = labelled['aug'] if 'aug' in labelled else False
    train = (labelled.fold != held_out_fold).values
    test = ((labelled.fold == held_out_fold) & ~is_copy).values

    scaler = StandardScaler().fit(embeddings[train])
    probe = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced')
    probe.fit(scaler.transform(embeddings[train]), labels[train])
    scores = metrics(labels[test], probe.predict_proba(scaler.transform(embeddings[test])))

    model.train()
    return scores['macro_f1'], scores['dialect_f1_on_dialect_clips']


# -------------------------------------------------------------------------------------- training
def pretrain_encoder(
    run_name,
    make_model,
    objectives,
    steps=12000,
    batch_size=48,
    learning_rate=1e-3,
    sources=('vizina', 'audiomoth_bhbl', 'birdnet_yh'),
    other_song_probability=0.0,
    teacher_sees_clean_audio=False,
    codebook=None,
    noise_bank='data/noise_bank_clean.npy',
    speed_jitter=0.02,
    temperature=0.1,
    ema_decay=0.999,
    probe_every=2000,
    threads=10,
    data_workers=3,
):
    """Pretrain an encoder without labels.

    Args:
        run_name: output directory under `$STRNADI_WS/ssl/`.
        make_model: function returning a StrnadNet-like model (it must offer `frontend`, `frames`,
            `pool` and `emb`).
        objectives: {objective name: weight}; see the module docstring for the names.
        other_song_probability: chance that the second view is a different song of the same
            source. Keep at 0 for dialect work (see the module docstring).
        teacher_sees_clean_audio: for `masked_frames`, give the teacher the un-augmented song
            (denoising variant) instead of the student's view.
        codebook: for `masked_units`, the .npz written by `evaluation/vq_tokens.py fit`.
        noise_bank: background-noise bank for the views (skipped if the file is missing).
    """
    unknown = set(objectives) - set(OBJECTIVES)
    if unknown:
        raise ValueError(f'unknown objectives {sorted(unknown)}; choose from {OBJECTIVES}')
    weight = {name: float(objectives.get(name, 0.0)) for name in OBJECTIVES}

    model = make_model()
    print(f'{type(model).__name__}: {count_parameters(model):,} parameters', flush=True)
    if dry_run():
        frames = model.frames(torch.randn(2, WINDOW_SAMPLES) * 0.1)
        print(f'[dry run] {run_name}: frames {tuple(frames.shape)}, objectives {objectives}')
        return

    torch.set_num_threads(threads)
    out_dir = f'{WS}/ssl/{run_name}'
    os.makedirs(out_dir, exist_ok=True)

    # ---- data
    songs = load_pretraining_songs(list(sources))
    noise = None
    if os.path.exists(workspace_path(noise_bank)):
        noise = list(np.load(workspace_path(noise_bank), allow_pickle=True))
    dataset = SongViews(songs, noise, speed_jitter, other_song_probability)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=data_workers,
        persistent_workers=data_workers > 0,
        prefetch_factor=4 if data_workers > 0 else None,
    )
    batches = iter(loader)
    labelled = read_table('data/orig_index.csv')

    # ---- model, EMA teacher and the small heads each objective needs
    model = model.to(memory_format=torch.channels_last)
    teacher = copy.deepcopy(model)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    frame_dim = model.proj[0].out_channels
    embedding_dim = model.emb[0].out_features

    contrast_projection = nn.Sequential(
        nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, 128)
    )
    frame_predictor = nn.Conv1d(frame_dim, frame_dim, 1)
    source_classifier = nn.Sequential(
        nn.Linear(embedding_dim, 128, bias=False), nn.Linear(128, len(dataset.sources), bias=False)
    )
    span_decoder = nn.Sequential(
        nn.Conv1d(frame_dim, 256, 3, padding=1),
        nn.GELU(),
        nn.Conv1d(256, TARGET_BINS, 3, padding=1),
    )
    bottleneck_decoder = BottleneckDecoder(embedding_dim)

    unit_centroids, unit_encoder, unit_classifier = None, None, None
    if weight['masked_units'] > 0:
        codebook_file = np.load(workspace_path(codebook), allow_pickle=True)
        unit_centroids = torch.from_numpy(codebook_file['centroids'])
        encoder_path = str(codebook_file['encoder']) if 'encoder' in codebook_file else ''
        if encoder_path:
            # Units were fitted on the features of an earlier encoder (HuBERT iteration >= 2).
            state = torch.load(encoder_path, map_location='cpu', weights_only=False)['state']
            unit_encoder = load_encoder_state(make_model(), state).eval()
        unit_classifier = nn.Conv1d(frame_dim, len(unit_centroids), 1)

    parameters = list(model.parameters())
    parameters += list(contrast_projection.parameters()) + list(frame_predictor.parameters())
    if weight['source_prediction'] > 0:
        parameters += list(source_classifier.parameters())
    if weight['masked_reconstruction'] > 0:
        parameters += list(span_decoder.parameters())
    if weight['bottleneck_reconstruction'] > 0:
        parameters += list(bottleneck_decoder.parameters())
    if weight['masked_units'] > 0:
        parameters += list(unit_classifier.parameters())

    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.05)

    def lr_factor(step):
        warmup = min(1.0, (step + 1) / 500)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))
        return warmup * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    def spectrogram_of(waveforms):
        return model.frontend(waveforms).contiguous(memory_format=torch.channels_last)

    # ---- training loop
    log = []
    started = time.time()
    for step in range(steps):
        first_view, second_view, source_ids, clean_view = next(batches)
        first_spec = spectrogram_of(first_view)
        second_spec = spectrogram_of(second_view)

        loss = 0
        parts = {}

        # Clip embeddings of both views, shared by the objectives that need them.
        embeddings = None
        needs_embeddings = weight['same_song'] > 0 or weight['source_prediction'] > 0
        if needs_embeddings:
            frames = model.frames(torch.cat([first_spec, second_spec]))
            embeddings = clip_embedding(model, frames)
            both_source_ids = torch.cat([source_ids, source_ids])

        if weight['same_song'] > 0:
            projected = contrast_projection(embeddings)
            parts['same_song'] = supcon(projected, both_source_ids, temperature)

        if weight['source_prediction'] > 0:
            source_logits = source_classifier(embeddings)
            parts['source_prediction'] = F.cross_entropy(source_logits, both_source_ids.long())

        if weight['masked_frames'] > 0:
            with torch.no_grad():
                teacher_input = (
                    spectrogram_of(clean_view) if teacher_sees_clean_audio else second_spec
                )
                teacher_frames = F.instance_norm(teacher.frames(teacher_input))
            masked_spec, hidden = mask_time_spans(second_spec)
            predicted = frame_predictor(model.frames(masked_spec))
            frame_mask = mask_at_frame_rate(hidden, predicted.shape[-1])
            per_frame = F.smooth_l1_loss(predicted, teacher_frames, reduction='none').mean(1)
            parts['masked_frames'] = masked_mean(per_frame, frame_mask)

        needs_target = (
            weight['masked_reconstruction'] > 0 or weight['bottleneck_reconstruction'] > 0
        )
        if needs_target:
            target = reconstruction_target(second_spec)

        if weight['masked_reconstruction'] > 0:
            masked_spec, hidden = mask_time_spans(second_spec, ratio=0.4, span=16)
            reconstruction = span_decoder(model.frames(masked_spec))
            reconstruction = F.interpolate(reconstruction, size=TARGET_FRAMES)
            frame_mask = mask_at_frame_rate(hidden, TARGET_FRAMES)
            per_frame = ((reconstruction - target) ** 2).mean(1)
            parts['masked_reconstruction'] = masked_mean(per_frame, frame_mask)

        if weight['masked_units'] > 0:
            with torch.no_grad():
                clean_spec = spectrogram_of(clean_view)
                if unit_encoder is not None:
                    unit_features = unit_encoder.frames(clean_spec).transpose(1, 2)
                else:
                    # First iteration: units of the 40 ms log-mel frames themselves.
                    unit_features = (
                        F.avg_pool2d(clean_spec.float(), (1, 4)).squeeze(1).transpose(1, 2)
                    )
                mean = unit_features.mean(-1, keepdim=True)
                std = unit_features.std(-1, keepdim=True) + 1e-5
                unit_features = (unit_features - mean) / std
                centroids = unit_centroids[None].expand(len(unit_features), -1, -1)
                units = torch.cdist(unit_features, centroids).argmin(-1)  # (B, frames)

            masked_spec, hidden = mask_time_spans(second_spec, ratio=0.4, span=12)
            unit_logits = unit_classifier(model.frames(masked_spec))  # (B, units, frames)
            n_frames = unit_logits.shape[-1]
            frame_mask = mask_at_frame_rate(hidden, n_frames)
            per_frame = F.cross_entropy(unit_logits, units[:, :n_frames], reduction='none')
            parts['masked_units'] = masked_mean(per_frame, frame_mask)

        if weight['bottleneck_reconstruction'] > 0:
            if embeddings is not None:
                second_embedding = embeddings[len(first_spec) :]
            else:
                second_embedding = clip_embedding(model, model.frames(second_spec))
            parts['bottleneck_reconstruction'] = F.mse_loss(
                bottleneck_decoder(second_embedding), target
            )

        for name, value in parts.items():
            loss = loss + weight[name] * value

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 3.0)
        optimizer.step()
        scheduler.step()

        # The teacher follows the student as an exponential moving average.
        with torch.no_grad():
            student_state = model.state_dict()
            for name, averaged in teacher.state_dict().items():
                if averaged.dtype.is_floating_point:
                    averaged.mul_(ema_decay).add_(student_state[name].detach(), alpha=1 - ema_decay)
                else:
                    averaged.copy_(student_state[name])

        values = {name: round(value.item(), 4) for name, value in parts.items()}
        if step % 50 == 0:
            print(step, values, round(time.time() - started), flush=True)

        is_last_step = step == steps - 1
        if (step + 1) % probe_every == 0 or is_last_step:
            probe_scores = linear_probe(teacher, labelled)
            entry = {'step': step + 1, 'probe': probe_scores, 't': time.time() - started}
            entry.update(values)
            log.append(entry)
            print('PROBE', entry, flush=True)

            save_model(f'{out_dir}/encoder.pt', teacher.state_dict(), model, epoch=step + 1)
            with open(f'{out_dir}/log.json', 'w') as file:
                json.dump(log, file, indent=1)

    # Marker file: fine-tuning jobs queued behind this run wait for it.
    with open(f'{out_dir}/done', 'w') as file:
        file.write('ok')
