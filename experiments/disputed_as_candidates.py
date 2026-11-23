# %% [markdown]
# # disputed_as_candidates
#
# Train the disputed dialect clips as candidate sets {label, Unfinished} instead of hard labels;
# evaluate on the original labels.
#
# **Question.** About 310 errors are shared by four independently trained recipes, and in 163
# clips their average confidently contradicts the label (mostly "dialect" clips showing a phrase
# and ONE terminal element, which the project's own rule calls Unfinished). Training on them as
# hard dialect labels teaches the model to call incomplete songs a dialect. Until experts re-label
# them, a candidate set is the honest target.
#
# **Status:** proposed (from review proposal). Not run. Sensitivity bound (circular, not a
# result): treating the confident disagreements as label errors moves macro-F1 from 0.78 to
# 0.85-0.88.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv; processing/build_disputed.py (-> data/orig_aligned_disputed.csv with a
# `train_label` column)
#
# Run from the repository root with `python -m experiments.disputed_as_candidates`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/runs/disputed_as_candidates/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'disputed_as_candidates'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned_disputed.csv',
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
