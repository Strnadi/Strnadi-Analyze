# %% [markdown]
# # bc_resnet_tiny
#
# BC-ResNet (tau = 2, ~0.15 M params): the published tiny-model family from keyword spotting /
# DCASE low-complexity.
#
# **Question.** How small can the phone model get? StrnadNet is a home-made design of the same
# kind that was never compared against the proven family.
#
# **Status:** proposed (from review proposal). Not run.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.bc_resnet_tiny`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/bc_resnet_tiny/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.bc_resnet import BCResNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'bc_resnet_tiny'

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
)


# %% Model
def make_model():
    return BCResNet(tau=2.0, head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
