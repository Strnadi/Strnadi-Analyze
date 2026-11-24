# %% [markdown]
# # perch_frozen_temporal
#
# Frozen Perch 2.0 spatial embedding kept as a 16-step time series -> temporal conv + attentive
# pooling head.
#
# **Question.** Keeping Perch's time axis recovers the element order that pooling throws away.
#
# **Status:** rerun (from the research session). Re-targeted on 2026-11-24 to the audited clips and
# best-validation-loss checkpoints, 5 folds; results in docs/RESULTS.md. 0.741 macro-F1, acc 0.863 (pooled MLP: 0.691):
# equal to the from-scratch StrnadNet baseline (0.745) with no fine-tuning, and a fold trains in
# minutes. BirdNET 3.0 under the same protocol: 0.523.
#
# **Needs.** emb/orig_perch.npz
#
# Run from the repository root with `python -m experiments.perch_frozen_temporal`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/perch_frozen_temporal/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.perch_temporal import PerchTemporal
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'perch_frozen_temporal'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    inputs='perch_spatial',
    eval_window='center',
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
    return PerchTemporal(precomputed=True)


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
