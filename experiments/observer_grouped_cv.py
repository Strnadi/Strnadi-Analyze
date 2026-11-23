# %% [markdown]
# # observer_grouped_cv
#
# Robustness check of the protocol: folds grouped by observer (and duplicate-audio groups) instead
# of by recording.
#
# **Question.** Recording-grouped folds do not separate observers, sites or sessions: knowing the
# observer predicts the dialect with 0.81 accuracy (majority class 0.69), and 107 groups of byte-
# identical NCS recordings sit under different F-ids (81 span folds, some with conflicting
# labels). If scores drop under observer grouping, the model leans on recordist / device / site.
#
# **Status:** proposed (from review proposal). Not run.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv; processing/make_folds.py --group observer --dedupe (->
# data/orig_aligned_observer_folds.csv)
#
# Run from the repository root with `python -m experiments.observer_grouped_cv`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/observer_grouped_cv/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'observer_grouped_cv'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned_observer_folds.csv',
    extra_tables={
        'data/weak_v3.csv': 0.35,
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
