# %% [markdown]
# # fdy_crnn
#
# FDY-CRNN: frequency-dynamic convolutions + BiGRU, the standard strong sound-event-detection
# network.
#
# **Question.** Kernels that adapt along frequency fit absolute-pitch patterns better than one
# shared kernel plus a coordinate channel.
#
# **Status:** rerun (from review proposal). Re-targeted on 2026-11-24 to the audited clips, the clean
# noise bank and best-validation-loss checkpoints, 5 folds; results in docs/RESULTS.md. Not run.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.fdy_crnn`, or step through the cells in
# an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/fdy_crnn/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.fdy_crnn import FDYCRNN
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'fdy_crnn'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    eval_window='center',
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
    return FDYCRNN(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
