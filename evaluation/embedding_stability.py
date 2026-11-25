"""Do two trainings put the classes in the same places of the embedding?

    python -m evaluation.embedding_stability <run> <the same recipe, another seed>

For the held-out clips of every fold, embedded by the fold models of both runs:

    class centres   cosine between the two runs' mean embedding of every class, with no alignment
                    of the two spaces (1 = the same direction, about 0 = unrelated)
    same clip       cosine between the two runs' embeddings of the same clip
    fixed map       the place of every clip on the fixed 2-D read-out
                    (`losses.fixed_prototypes.map_coordinates`): how far apart the two runs put
                    the same clip (the dialect anchors lie on a circle of radius 1), and how well
                    the map separates the classes (5-nearest-neighbour accuracy, leave one out)

An ordinary run scores about 0 on the first two: its classes are separated, but in directions
that its random start chose. A run trained with `prototype_loss_weight` should score near 1, and
its two maps should look alike (`figs/embedding_stability_<run>.png`).
"""

import argparse
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CLASSES, WS, effective_run_name
from evaluation.embedding_map import held_out_embeddings, leave_one_out_knn_accuracy
from losses.fixed_prototypes import MAP_ANCHORS, map_coordinates

COLOURS = {
    'BC': '#d4b400',
    'BE': '#24581a',
    'BhBl': '#61c1da',
    'BlBh': '#19268e',
    'XB': '#e2231a',
    'Unfinished': '#9a9a9a',
    'None': '#000000',
}


def unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True).clip(1e-6)


def fold_pairs(run_a, run_b, device):
    """For every fold: the held-out embeddings of the same clips from both runs, and the labels.

    The folds are kept apart: every fold model has its own embedding space, and fold k of one
    seed starts from other random weights than fold k of the other (two models that start from
    the same weights end up with similar spaces even on different data).
    """
    pairs = []
    for fold in range(5):
        embeddings_a, clips_a = held_out_embeddings(run_a, fold, device)
        embeddings_b, clips_b = held_out_embeddings(run_b, fold, device)
        names_a = np.asarray(clips_a.name, dtype=str)
        row_of = {name: row for row, name in enumerate(np.asarray(clips_b.name, dtype=str))}
        rows_a = [row for row, name in enumerate(names_a) if name in row_of]
        rows_b = [row_of[names_a[row]] for row in rows_a]
        labels = np.asarray(clips_a.label, dtype=str)[rows_a]
        pairs.append((embeddings_a[rows_a], embeddings_b[rows_b], labels))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run_a')
    parser.add_argument('run_b')
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pairs = fold_pairs(arguments.run_a, arguments.run_b, device)

    centre_cosines = {name: [] for name in CLASSES}
    clip_cosines = []
    for fold_a, fold_b, fold_labels in pairs:
        unit_a, unit_b = unit(fold_a), unit(fold_b)
        clip_cosines.append((unit_a * unit_b).sum(1))
        for name in CLASSES:
            of_class = fold_labels == name
            if of_class.any():
                centre_a, centre_b = unit(unit_a[of_class].mean(0)), unit(unit_b[of_class].mean(0))
                centre_cosines[name].append(float(centre_a @ centre_b))
    centre_cosines = {name: float(np.mean(values)) for name, values in centre_cosines.items()}
    embeddings_a = np.concatenate([fold_a for fold_a, _, _ in pairs])
    embeddings_b = np.concatenate([fold_b for _, fold_b, _ in pairs])
    labels = np.concatenate([fold_labels for _, _, fold_labels in pairs])
    map_a, map_b = map_coordinates(embeddings_a), map_coordinates(embeddings_b)
    anchors = np.array([MAP_ANCHORS[name] for name in CLASSES])
    nearest_anchor = np.array(CLASSES)[((map_a[:, None] - anchors[None]) ** 2).sum(-1).argmin(1)]

    result = {
        'runs': [arguments.run_a, arguments.run_b],
        'n_clips': len(labels),
        'class_centre_cosine_between_runs': centre_cosines,
        'class_centre_cosine_mean': float(np.mean(list(centre_cosines.values()))),
        'same_clip_cosine_between_runs': float(np.concatenate(clip_cosines).mean()),
        'fixed_map': {
            'median_distance_between_runs': float(np.median(np.linalg.norm(map_a - map_b, axis=1))),
            'knn_accuracy_run_a': leave_one_out_knn_accuracy(map_a, labels),
            'knn_accuracy_run_b': leave_one_out_knn_accuracy(map_b, labels),
            'clips_nearest_to_their_own_anchor': float((nearest_anchor == labels).mean()),
        },
        'knn_accuracy_embedding_fold0_run_a': leave_one_out_knn_accuracy(pairs[0][0], pairs[0][2]),
    }
    print(json.dumps(result, indent=1))
    run = effective_run_name(arguments.run_a)
    with open(f'{WS}/results/embedding_stability_{run}.json', 'w') as file:
        json.dump(result, file, indent=1)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
    for ax, points, title in zip(axes, (map_a, map_b), (arguments.run_a, arguments.run_b)):
        for name in CLASSES:
            of_class = labels == name
            ax.scatter(
                points[of_class, 0], points[of_class, 1], s=5, color=COLOURS[name], label=name
            )
        for name, (x, y) in MAP_ANCHORS.items():
            ax.plot(x, y, marker='+', color='k', ms=10)
        ax.set_title(f'{title}: fixed read-out of the embedding', fontsize=10, loc='left')
        ax.set_aspect('equal')
    axes[0].legend(markerscale=3, fontsize=9, frameon=False)
    figure.tight_layout()
    figure.savefig(f'{WS}/figs/embedding_stability_{run}.png', dpi=80)
    print(f'{WS}/figs/embedding_stability_{run}.png')


if __name__ == '__main__':
    main()
