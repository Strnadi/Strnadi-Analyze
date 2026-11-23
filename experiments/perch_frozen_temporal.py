from .recipes import RESEARCH_PROTOCOL

EXPERIMENT = dict(
    RESEARCH_PROTOCOL, model='perch_temporal', model_kw=dict(precomputed=True), inputs='perch_spatial', lr=1e-3, epochs=30, epoch_size=8192, bs=64,
    description='Frozen Perch 2.0 spatial embedding kept as a 16-step time series -> temporal conv + attentive pooling head.',
    hypothesis='Keeping Perch\'s time axis recovers the element order that pooling throws away.',
    origin='research-session', status='done',
    result='0.741 macro-F1, acc 0.863 (pooled MLP: 0.691): equal to the from-scratch StrnadNet baseline (0.745) with no '
           'fine-tuning, and a fold trains in minutes. BirdNET 3.0 under the same protocol: 0.523.',
    needs='emb/orig_perch.npz',
)
