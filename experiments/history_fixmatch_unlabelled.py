# %% [markdown]
# # history_fixmatch_unlabelled
#
# *Run of the research session on the ORIGINAL clip labels, kept for the record. Its stored
# predictions carry the run name given in `docs/RESEARCH_LOG.md`.*
#
# The weak-pool recipe with the factorised head, plus FixMatch / FreeMatch consistency on 77.5k unlabelled songs (EMA teacher, adaptive
# thresholds).
#
# **Question.** Semi-supervised training inside the main loop is worth more than separate
# pretraining at this scale.
#
# **Status:** done (from the research session). 5 folds: macro-F1 0.775 (factorised head without it: 0.774), difference
# -0.001 [-0.015, +0.013]. The run without FixMatch is ahead on BhBl (+0.035) and BlBh (+0.051) recall; expert bird-level
# accuracy 0.924 (without FixMatch: 0.939). FixMatch gives no gain at this scale and was not adopted (research log
# 17.5). The review caveat explains why: the pseudo-labels come from the model itself. Retry only
# with a strong teacher (distill_function_matching.py).
#
# Run from the repository root with `python -m experiments.history_fixmatch_unlabelled`, or step through the cells
# in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/history_fixmatch_unlabelled/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'history_fixmatch_unlabelled'

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
