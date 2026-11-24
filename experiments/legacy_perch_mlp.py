# %% [markdown]
# # legacy_perch_mlp
#
# Legacy Perch experiment, ported: frozen Perch 2.0 pooled embedding (1536-d) -> 512 -> classes.
#
# **Question.** A bird foundation model's embedding is enough for dialects.
#
# **Status:** done (from the original Keras repository). Research session, same protocol: logistic
# 0.662, MLP 0.691 macro-F1; BhBl recall 0.47-0.51, BlBh 0.42-0.59. Few-shot: 0.36 -> 0.58
# macro-F1 for 4 -> 64 shots. 2-D maps: kNN dialect-F1 <= 0.42, HDBSCAN finds no clusters (also
# the outcome of the legacy t-SNE script; evaluation/research_session/map2d.py replaces it). Pooling discards
# element order: see perch_frozen_temporal.py.
#
# **Needs.** emb/orig_perch.npz (processing/extract_foundation_emb.py)
#
# Run from the repository root with `python -m experiments.legacy_perch_mlp`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/legacy_perch_mlp/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.perch_mlp import PerchMLP
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'legacy_perch_mlp'

settings = TrainSettings(
    # data
    labelled_table='data/orig_index.csv',
    inputs='perch_pooled',
    eval_window='start',
    # protocol
    checkpoint_rule='inner_argmax',
    # optimisation
    epochs=30,
    samples_per_epoch=8192,
    batch_size=64,
    learning_rate=0.001,
    # waveform augmentation
    snr_reference='clip',
)


# %% Model
def make_model():
    return PerchMLP(precomputed=True)


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
