# %% [markdown]
# # history_hard_negatives_codec
#
# R5: R2 + 6,865 other-species windows as "None" + phone-codec round-trips (AAC / Opus / MP3) on
# 15 % of clips.
#
# **Question.** Fewer false alarms on similar trills (Chiffchaff, Chaffinch, ...) and robustness
# to app audio compression.
#
# **Status:** done (from the research session). 5 folds: macro-F1 0.761, acc 0.867; -0.013 against
# R2, P(R2 better) 0.95. None recall +0.048 (the hard negatives work), dialects worse (BC -0.015, XB
# -0.022, BhBl -0.035). To be retried with a lower share of hard negatives; the codec part cannot
# show a benefit without phone audio (research log 17.5, device_robustness.py).
#
# Run from the repository root with `python -m experiments.history_hard_negatives_codec`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/history_hard_negatives_codec/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'history_hard_negatives_codec'

settings = TrainSettings(
    # data
    labelled_table='data/orig_index.csv',
    extra_tables={
        'data/weak_v3.csv': 0.33,
        'data/cf_v3.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
        'data/hardneg_v1.csv': 0.05,
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
    codec_probability=0.15,
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
