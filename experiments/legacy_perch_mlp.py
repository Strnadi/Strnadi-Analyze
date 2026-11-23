from .recipes import RESEARCH_PROTOCOL

EXPERIMENT = dict(
    RESEARCH_PROTOCOL, model='perch_mlp', model_kw=dict(precomputed=True), inputs='perch_pooled', lr=1e-3, epochs=30, epoch_size=8192, bs=64,
    description='Legacy Perch experiment, ported: frozen Perch 2.0 pooled embedding (1536-d) -> 512 -> classes.',
    hypothesis='A bird foundation model\'s embedding is enough for dialects.',
    origin='legacy-repo', status='done',
    result='Research session, same protocol: logistic 0.662, MLP 0.691 macro-F1; BhBl recall 0.47-0.51, BlBh 0.42-0.59. Few-shot: '
           '0.36 -> 0.58 macro-F1 for 4 -> 64 shots. 2-D maps: kNN dialect-F1 <= 0.42, HDBSCAN finds no clusters (also the outcome '
           'of the legacy t-SNE script; evaluation/map2d.py replaces it). Pooling discards element order: see perch_frozen_temporal.py.',
    needs='emb/orig_perch.npz (processing/extract_foundation_emb.py)',
)
