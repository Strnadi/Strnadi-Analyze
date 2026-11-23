# %% [markdown]
# # perch_distill_cached
#
# R7-clean + embedding distillation: the student's 128-d embedding is projected to Perch 2.0's
# 1536-d space and pulled towards the cached Perch embedding of the clean clip (cosine loss).
#
# **Question.** Perch's embedding space carries what 1.5 M recordings taught it; matching it
# regularises the small model. This is the BirdCLEF 2026 winning recipe and gives every BioAcu
# project head one shared embedding.
#
# **Status:** proposed (from the research session). Implemented in the research trainer
# (--w_distill) but never queued. Rows without a cached Perch embedding are masked out of the
# distillation term.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv; emb/orig_perch.npz, and emb/rows_perch_emb.npy for the weak tables
#
# Run from the repository root with `python -m experiments.perch_distill_cached`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/perch_distill_cached/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'perch_distill_cached'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    eval_window='center',
    # protocol
    checkpoint_rule='inner_smooth3',
    # optimisation
    epochs=25,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
    # weak-label and auxiliary losses
    distill_weight=0.5,
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
