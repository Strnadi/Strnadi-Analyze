# %% [markdown]
# # r4_fixmatch
#
# R4: R2 + FixMatch / FreeMatch consistency on 77.5k unlabelled songs (EMA teacher, adaptive
# thresholds).
#
# **Question.** Semi-supervised training inside the main loop is worth more than separate
# pretraining at this scale.
#
# **Status:** running (from the research session). In progress on 2026-11-23 (first attempt
# crashed at epoch 10 on an exhausted loader; fixed). Fold 0/1 outer F1 after 20 epochs: 0.80 /
# 0.76, i.e. inside the fold-to-fold spread of R2. Review caveat: the pseudo-labels come from the
# model itself, 78 % of unlabelled passive songs are predicted Unfinished, and BirdCLEF teams
# found weak self-teachers hurt.
#
# Run from the repository root with `python -m experiments.r4_fixmatch`, or step through the cells
# in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/r4_fixmatch/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'r4_fixmatch'

settings = TrainSettings(
    # data
    labelled_table='data/orig_index.csv',
    extra_tables={
        'data/weak_v3.csv': 0.35,
        'data/cf_v3.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    unlabelled_table='data/unlabeled_v2.csv',
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
    # runtime
    data_workers=3,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
