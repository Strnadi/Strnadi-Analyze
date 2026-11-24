"""A 2-D map of a model's clip embeddings: can an expert "draw a circle around a dialect"?

    python -m evaluation.embedding_map <run> [--fold 0]

The BioAcu platform wants a map on which an expert outlines a dialect by hand. On frozen,
pooled Perch 2.0 embeddings that does not work for Yellowhammer dialects (research log, section 8:
silhouette ~ 0, HDBSCAN finds no clusters, kNN dialect-F1 in 2-D 0.15-0.42). This script asks the
same questions of a model trained for the task.

Only held-out clips are used: the fold model embeds the clips of its own test fold, whose
recordings it never saw. Reported:

    kNN accuracy (leave-one-out, k = 5) in the full embedding and in the 2-D map
    silhouette of the classes in both spaces
    HDBSCAN clusters found on the 2-D map without any labels, and how pure they are

and `figs/embedding_map_<run>_fold<k>.png`: the map coloured by label, and by HDBSCAN cluster.
"""

import argparse
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from config import CLASSES, WS, effective_run_name
from evaluation.report import load_run_predictions
from training.checkpoint import load_model
from training.data import load_seg
from training.inference import peak_normalize

CLASS_COLOURS = {
    'BC': '#e6c700',
    'BE': '#24581a',
    'BhBl': '#61c1da',
    'BlBh': '#19268e',
    'XB': '#fe0a04',
    'Unfinished': '#b0b0b0',
    'None': '#000000',
}


@torch.no_grad()
def held_out_embeddings(run, fold, device):
    """Embeddings (N, D) and the clip table of one fold's held-out clips."""
    predictions = load_run_predictions(run)
    clips = predictions[predictions.cv_fold == fold]
    model = load_model(f'{WS}/runs/{effective_run_name(run)}/fold{fold}/model.pt').to(device)

    waveforms = []
    for clip in clips.itertuples():
        start = clip.t0 + max((clip.t1 - clip.t0) - 4.0, 0) / 2
        waveforms.append(load_seg(clip.npy, start, start + 4.0, 128000))
    waveforms = torch.from_numpy(peak_normalize(np.stack(waveforms)))

    embeddings = []
    for first in range(0, len(waveforms), 128):
        batch = waveforms[first : first + 128].to(device)
        embeddings.append(model(batch)['emb'].cpu().numpy())
    return np.concatenate(embeddings), clips


def leave_one_out_knn_accuracy(points, labels, k=5):
    """Each point is classified by its k nearest OTHER points."""
    classifier = KNeighborsClassifier(n_neighbors=k + 1).fit(points, labels)
    neighbours = classifier.kneighbors(points, return_distance=False)[:, 1:]
    votes = labels[neighbours]
    predicted = [max(set(row), key=list(row).count) for row in votes]
    return float(np.mean(np.array(predicted) == labels))


def main():
    import umap

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run')
    parser.add_argument('--fold', type=int, default=0)
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings, clips = held_out_embeddings(arguments.run, arguments.fold, device)
    labels = np.asarray(clips.label, dtype=str)

    projection = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0)
    points = projection.fit_transform(embeddings)  # label-free: UMAP never sees the classes

    clusters = HDBSCAN(min_cluster_size=12).fit_predict(points)
    cluster_ids = [cluster for cluster in np.unique(clusters) if cluster >= 0]
    purities = []
    for cluster in cluster_ids:
        members = labels[clusters == cluster]
        purities.append(max(np.mean(members == name) for name in CLASSES))
    sizes = [int(np.sum(clusters == cluster)) for cluster in cluster_ids]

    result = {
        'run': effective_run_name(arguments.run),
        'fold': arguments.fold,
        'n_clips': len(clips),
        'knn_accuracy_embedding': leave_one_out_knn_accuracy(embeddings, labels),
        'knn_accuracy_2d': leave_one_out_knn_accuracy(points, labels),
        'silhouette_embedding': float(silhouette_score(embeddings, labels, metric='cosine')),
        'silhouette_2d': float(silhouette_score(points, labels)),
        'hdbscan_clusters': len(cluster_ids),
        'hdbscan_share_clustered': float(np.mean(clusters >= 0)),
        'hdbscan_purity_weighted': float(np.average(purities, weights=sizes)) if sizes else None,
    }
    print(json.dumps(result, indent=1))
    name = f'{result["run"]}_fold{arguments.fold}'
    with open(f'{WS}/results/embedding_map_{name}.json', 'w') as file:
        json.dump(result, file, indent=1)

    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 6.5))
    for class_name in CLASSES:
        of_class = labels == class_name
        left.scatter(
            points[of_class, 0],
            points[of_class, 1],
            s=9,
            color=CLASS_COLOURS[class_name],
            label=f'{class_name} ({int(of_class.sum())})',
        )
    left.legend(fontsize=8, markerscale=2)
    left.set_title(
        f'held-out clips by label (kNN accuracy in 2-D {result["knn_accuracy_2d"]:.2f})',
        fontsize=10,
    )
    right.scatter(points[:, 0], points[:, 1], s=9, c=clusters, cmap='tab20')
    right.set_title(
        f'the same map clustered without labels: {len(cluster_ids)} clusters, '
        f'purity {result["hdbscan_purity_weighted"]:.2f}',
        fontsize=10,
    )
    for ax in (left, right):
        ax.set_xticks([])
        ax.set_yticks([])
    figure.tight_layout()
    figure.savefig(f'{WS}/figs/embedding_map_{name}.png', dpi=90)
    print(f'saved {WS}/figs/embedding_map_{name}.png')


if __name__ == '__main__':
    main()
