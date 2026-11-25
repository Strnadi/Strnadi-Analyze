"""An embedding with fixed places: every class is pulled towards a direction that never changes.

A trained network puts the classes wherever its random start led it, so the embedding of two
trainings cannot be compared, and a map drawn from one (UMAP) looks different after the next.
Here every class owns a prototype direction that is a constant of the code: the same for every
seed, every fold and every later retraining. A cosine-margin loss (the CosFace / ArcFace family,
but with the class centres frozen instead of learned) pulls the embedding of a hand-labelled clip
towards its prototype. The classifier head is not touched; this is an extra loss on the embedding.

Because the prototypes are fixed, so is the map: `map_coordinates` is a fixed linear read-out of
the embedding onto a 2-D layout chosen once (`MAP_ANCHORS`). A new clip gets its place without
refitting anything, and a region an expert outlined on the map stays valid after retraining.
"""

import numpy as np
import torch
import torch.nn.functional as F

from config import CLASSES

#: Where every class sits on the map: the dialects on a circle, "Unfinished" in the middle (a
#: phrase with no dialect is equally far from all of them), "None" outside the circle.
MAP_ANCHORS = {
    'BC': (0.0, 1.0),
    'BE': (0.951, 0.309),
    'BhBl': (0.588, -0.809),
    'BlBh': (-0.588, -0.809),
    'XB': (-0.951, 0.309),
    'Unfinished': (0.0, 0.0),
    'None': (0.0, -1.7),
}


def fixed_prototypes(dim, n_classes=len(CLASSES)):
    """The class directions: (n_classes, dim), orthonormal, and the same on every call."""
    random = np.random.default_rng(20261125).normal(size=(dim, n_classes))
    orthonormal, _ = np.linalg.qr(random)  # (dim, n_classes) with orthonormal columns
    return torch.from_numpy(orthonormal.T.astype(np.float32).copy())


def prototype_loss(embedding, targets, is_hard, scale=16.0, margin=0.2):
    """Cosine-margin cross-entropy between the embedding and the fixed class prototypes.

    Args:
        embedding: (B, D) clip embeddings.
        targets: (B, n_classes) targets; only rows with a hard (one-hot) label are used.
        is_hard: (B,) bool, which rows carry a hard label.
        scale, margin: as in CosFace: logits = scale * (cos - margin on the target class).

    Returns:
        Scalar loss (zero, still connected to `embedding`, when the batch has no hard label).
    """
    if not is_hard.any():
        return embedding.sum() * 0.0
    prototypes = fixed_prototypes(embedding.shape[1]).to(embedding.device)
    cosines = F.normalize(embedding[is_hard].float(), dim=1) @ prototypes.T  # (H, n_classes)
    one_hot = targets[is_hard].float()
    return F.cross_entropy(scale * (cosines - margin * one_hot), one_hot.argmax(1))


def map_coordinates(embedding):
    """Embeddings (N, D) as numpy -> their place on the fixed map, (N, 2).

    The place is the average of the class anchors weighted by the cosine of the embedding to
    every class prototype: a clip that is all "BC" sits on the BC anchor, one between two classes
    between their anchors.
    """
    embedding = np.asarray(embedding, np.float32)
    unit = embedding / np.linalg.norm(embedding, axis=1, keepdims=True).clip(1e-6)
    cosines = unit @ fixed_prototypes(embedding.shape[1]).numpy().T  # (N, n_classes)
    anchors = np.array([MAP_ANCHORS[name] for name in CLASSES], np.float32)
    return cosines.clip(0) @ anchors
