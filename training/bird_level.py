"""Train the bird-level model: all songs of a recording -> one dialect.

    fit_bird_level(run_name, make_model, song_run='strnadnet_audited_labels')

Input is `results/song_emb_<song_run>.npz` from `evaluation/research_session/embed_songs.py`: per-song embeddings and
class probabilities that are already out of fold (each recording's songs were embedded by the fold
model that never saw that recording). The recording-level dialect is the bag label.

Reported, out of fold over recordings: accuracy of the bag model, next to the baseline it has to
beat (mean dialect probability over the songs the song model calls complete).
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from config import DIALECT_IDX, DIALECTS, WS, dry_run


def collect_bags(song_data):
    """Group songs by recording.

    Returns:
        list of (song indices, dialect index, cross-validation fold), one per recording whose
        label names exactly one dialect.
    """
    bags = []
    for recording in np.unique(song_data['rec']):
        song_indices = np.where(song_data['rec'] == recording)[0]
        dialect = song_data['bag_label'][song_indices[0]]
        if dialect in DIALECTS:
            fold = int(song_data['fold'][song_indices[0]])
            bags.append((song_indices, DIALECTS.index(dialect), fold))
    return bags


def make_batch(bags, embeddings, max_songs, rng=None):
    """Pad a list of bags to (B, max_songs, D) with a mask marking the real songs."""
    n_bags = len(bags)
    batch = np.zeros((n_bags, max_songs, embeddings.shape[1]), np.float32)
    mask = np.zeros((n_bags, max_songs), bool)
    labels = np.zeros(n_bags, np.int64)

    for position, (song_indices, dialect, _) in enumerate(bags):
        if len(song_indices) > max_songs:
            if rng is not None:
                song_indices = rng.choice(song_indices, max_songs, replace=False)
            else:
                song_indices = song_indices[:max_songs]
        batch[position, : len(song_indices)] = embeddings[song_indices]
        mask[position, : len(song_indices)] = True
        labels[position] = dialect

    return torch.from_numpy(batch), torch.from_numpy(mask), torch.from_numpy(labels)


def mean_probability_baseline(song_probabilities):
    """Baseline bird-level decision: average dialect probability over the songs called complete."""
    called_complete = np.isin(song_probabilities.argmax(1), DIALECT_IDX)
    if called_complete.any():
        dialect_probability = song_probabilities[called_complete][:, DIALECT_IDX].mean(0)
    else:
        dialect_probability = song_probabilities[:, DIALECT_IDX].mean(0)
    return int(dialect_probability.argmax())


def fit_bird_level(
    run_name,
    make_model,
    song_run,
    epochs=30,
    max_songs=24,
    batch_size=32,
    learning_rate=1e-3,
    threads=4,
):
    """Cross-validated training of the bird-level model.

    Args:
        run_name: output directory under `$STRNADI_WS/runs/`.
        make_model: function taking the embedding size and returning the bag model.
        song_run: the run whose song embeddings are used.
        max_songs: recordings with more songs are sub-sampled to this many per batch.
    """
    if dry_run():
        model = make_model(128)
        outputs = model(torch.randn(2, 5, 128), torch.ones(2, 5, dtype=torch.bool))
        print(f'[dry run] {run_name}: bag logits {tuple(outputs["logits"].shape)}')
        return None

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    song_data = np.load(f'{WS}/results/song_emb_{song_run}.npz', allow_pickle=True)
    embeddings = song_data['emb']
    bags = collect_bags(song_data)
    n_dialects = len(DIALECTS)

    out_dir = f'{WS}/runs/{run_name}'
    os.makedirs(out_dir, exist_ok=True)
    n_correct = 0
    n_correct_baseline = 0
    n_recordings = 0

    for fold in sorted({bag[2] for bag in bags}):
        train_bags = [bag for bag in bags if bag[2] != fold]
        test_bags = [bag for bag in bags if bag[2] == fold]

        model = make_model(embeddings.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

        # Square-root class weights: rare dialects count more, without dominating.
        counts = np.bincount([bag[1] for bag in train_bags], minlength=n_dialects)
        class_weights = (counts.sum() / np.maximum(counts, 1) / n_dialects) ** 0.5
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        for _ in range(epochs):
            model.train()
            order = rng.permutation(len(train_bags))
            for start in range(0, len(order) - batch_size + 1, batch_size):
                chosen = [train_bags[i] for i in order[start : start + batch_size]]
                batch, mask, labels = make_batch(chosen, embeddings, max_songs, rng)

                # The model's extra "no complete song" output stays unused until such bags are labelled.
                logits = model(batch, mask)['logits'][:, :n_dialects]
                loss = F.cross_entropy(logits, labels, weight=class_weights)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        torch.save({'state': model.state_dict()}, f'{out_dir}/fold{fold}.pt')

        with torch.no_grad():
            for start in range(0, len(test_bags), 64):
                batch, mask, labels = make_batch(
                    test_bags[start : start + 64], embeddings, max_songs
                )
                predicted = model(batch, mask)['logits'][:, :n_dialects].argmax(1)
                n_correct += int((predicted == labels).sum())

        for song_indices, dialect, _ in test_bags:
            baseline = mean_probability_baseline(song_data['prob'][song_indices])
            n_correct_baseline += int(baseline == dialect)

        n_recordings += len(test_bags)
        print(
            f'fold {fold}: bag model {n_correct / n_recordings:.3f}, '
            f'mean-probability baseline {n_correct_baseline / n_recordings:.3f} '
            f'(cumulative over {n_recordings} recordings)',
            flush=True,
        )

    result = {
        'name': run_name,
        'songs': song_run,
        'recordings': n_recordings,
        'bag_model_accuracy': n_correct / n_recordings,
        'mean_probability_accuracy': n_correct_baseline / n_recordings,
    }
    with open(f'{out_dir}/result.json', 'w') as file:
        json.dump(result, file, indent=1)
    print(json.dumps(result))
    return result
