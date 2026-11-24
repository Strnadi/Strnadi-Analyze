# %% [markdown]
# # contour_tcn
#
# Architecture comparison: 1-D TCN over top-3 pitch-contour features (0.27 M params, ~5x cheaper
# than StrnadNet).
#
# **Question.** The "ornithologist's view" (pitch tracks only) is enough, at a fraction of the
# cost.
#
# **Status:** rerun (from the research session). Re-targeted on 2026-11-24 to the audited clips, the clean
# noise bank and best-validation-loss checkpoints, 5 folds; results in docs/RESULTS.md. Not finished. See contour_ridge.py for the sub-
# bin variant.
#
# Run from the repository root with `python -m experiments.contour_tcn`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/contour_tcn/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.contour_tcn import ContourTCN
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'contour_tcn'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    eval_window='center',
    # optimisation
    epochs=25,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_reference='band',
    splice_probability=0.3,
)


# %% Model
def make_model():
    return ContourTCN(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
