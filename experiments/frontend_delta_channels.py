# %% [markdown]
# # frontend_delta_channels
#
# The spectrogram with its first and second time derivatives as extra channels.
#
# **Question.** The legacy MobileNet experiment fed [mel, delta, delta-delta] channels (and computed the deltas
# along frequency by mistake). Deltas mark onsets and offsets of elements. Do they help the
# reference network, with the derivative taken along time?
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `frontend=dict(norm='delta')` (dB spectrogram plus two derivative channels;
# this front-end has no per-frequency median removal).
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.frontend_delta_channels`.
# Training writes to `$STRNADI_WS/runs/frontend_delta_channels/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'frontend_delta_channels'

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
    snr_reference='band',
    splice_probability=0.3,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', frontend=dict(norm='delta'))


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
