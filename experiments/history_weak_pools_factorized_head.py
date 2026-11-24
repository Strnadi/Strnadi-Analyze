# %% [markdown]
# # history_weak_pools_factorized_head
#
# *Run of the research session on the ORIGINAL clip labels, kept for the record. Its stored
# predictions carry the run name given in `docs/RESEARCH_LOG.md`.*
#
# The weak pools with the factorised head. Weak file-level labels train only the dialect factor, weighted by
# p(complete).
#
# **Question.** Weak labels help once they can no longer drag hard songs into "Unfinished" (the
# cheapest way to satisfy "BC or Unfinished or None").
#
# **Status:** done (from the research session). 5 folds: macro-F1 0.774, acc 0.874; recall BhBl
# 0.55, BlBh 0.65, None 0.50, Unfinished 0.92, XB 0.95. vs base_v3 +0.029 [+0.013, +0.046] (paired
# recording bootstrap, one seed each); vs `history_weak_pools_flat_head` +0.020 [+0.007, +0.034]. Expert bird-level BE/BlBh
# accuracy 0.939 (base 0.893, original NCS file labels 0.863). Trained with three since-fixed
# defects: black-patch counterfactual filler (cf_v3), clip-power SNR, 20 % of strong clips
# starting mid-song. Complete-vs-Unfinished AUC: BhBl 0.83, BlBh 0.89, BC 0.93, BE 0.95, XB 0.98.
#
# Run from the repository root with `python -m experiments.history_weak_pools_factorized_head`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/history_weak_pools_factorized_head/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'history_weak_pools_factorized_head'

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
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
