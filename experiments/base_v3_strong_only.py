# %% [markdown]
# # base_v3_strong_only
#
# Baseline: StrnadNet (0.9 M params) on the 3,744 human-labelled clips only, recording-grouped
# 5-fold CV.
#
# **Question.** Reference point every other recipe has to beat on the same folds.
#
# **Status:** done (from the research session). 5 folds: macro-F1 0.745, acc 0.868; recall BC
# 0.82, BE 0.85, BhBl 0.50, BlBh 0.49, None 0.43, Unfinished 0.93, XB 0.93. Dialect accuracy given
# a complete song 0.974; BhBl<->BlBh confusions 1 of 223; 69 of 144 BhBl clips predicted
# Unfinished. Held-out Vizina males (aligned clips): BhBl recall 0.96 vs 0.68 for the production
# model. NCS bird-level accuracy 0.957 at coverage 0.89. ECE 0.020. Training loss reaches the
# label-smoothing floor by epoch ~12 while held-out F1 is flat from epoch 4-8: the labelled set is
# memorised.
#
# Run from the repository root with `python -m experiments.base_v3_strong_only`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/base_v3_strong_only/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'base_v3_strong_only'

settings = TrainSettings(
    # data
    labelled_table='data/orig_index.csv',
    eval_window='start',
    # protocol
    checkpoint_rule='inner_argmax',
    # optimisation
    epochs=25,
    samples_per_epoch=4096,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    snr_reference='clip',
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
