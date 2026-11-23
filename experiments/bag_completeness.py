# %% [markdown]
# # bag_completeness
#
# R7-clean + recording-level bag loss: of 6 songs drawn from an NCS recording filed as dialect D,
# at least one must be a complete song of D (noisy-OR over p(song) p(complete) q(D)).
#
# **Question.** With the factorised head the 45k weak songs teach nothing about completeness, the
# model's actual weakness. The bag constraint supplies that signal: some song in the recording has
# to be called complete.
#
# **Status:** proposed (from review proposal). Not run. The constraint was in the research
# session's first plan ("at least one song in the recording is D") and was never implemented. Not
# applied to site-level labels (Vizina / AudioMoth / JCU).
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.bag_completeness`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/bag_completeness/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'bag_completeness'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v3.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    bag_table='data/weak_v3.csv',
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
    bag_loss_weight=0.5,
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
