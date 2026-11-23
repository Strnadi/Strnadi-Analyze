# %% [markdown]
# # arch_strnadnet_wide
#
# Architecture comparison: StrnadNet twice as wide (3.5 M params) on the R7 recipe.
#
# **Question.** Is 0.9 M parameters holding accuracy back?
#
# **Status:** queued (from the research session). Not finished. Prior from the learning curves:
# capacity is not the limit (the 0.9 M model already memorises the labelled set and held-out F1 is
# flat from epoch 4-8).
#
# Run from the repository root with `python -m experiments.arch_strnadnet_wide`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/arch_strnadnet_wide/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'arch_strnadnet_wide'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_audiomoth.npy',
    eval_window='start',
    # protocol
    folds=[0, 1, 2],
    checkpoint_rule='inner_argmax',
    # optimisation
    epochs=25,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_reference='band',
    splice_probability=0.3,
)


# %% Model
def make_model():
    return StrnadNet(widths=(32, 64, 128, 256), d=256, head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
