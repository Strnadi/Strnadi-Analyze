# %% [markdown]
# # history_weak_pools_flat_head
#
# R1: strong clips + 47.8k weak songs as candidate sets {file dialect, Unfinished, None} +
# counterfactual songs + Vizina annotations, real AudioMoth noise, null splices. Flat 7-way head.
#
# **Question.** Does the big weakly labelled pool fix the citizen-recording errors (complete songs
# called Unfinished)?
#
# **Status:** done (from the research session). 5 folds: macro-F1 0.754; vs base_v3 +0.009
# [-0.009, +0.026], not significant. The weak data only pays off with the factorised head (R2).
#
# Run from the repository root with `python -m experiments.history_weak_pools_flat_head`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/history_weak_pools_flat_head/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'history_weak_pools_flat_head'

settings = TrainSettings(
    # data
    labelled_table='data/orig_index.csv',
    extra_tables={
        'data/weak_v3.csv': 0.35,
        'data/cf_v3.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_audiomoth.npy',
    eval_window='start',
    # protocol
    checkpoint_rule='inner_argmax',
    # optimisation
    epochs=25,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_reference='clip',
    splice_probability=0.3,
)


# %% Model
def make_model():
    return StrnadNet()


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
