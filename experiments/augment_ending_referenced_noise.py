# %% [markdown]
# # augment_ending_referenced_noise
#
# Noise level set relative to the song ending, 6-26 dB.
#
# **Question.** Noise mixed relative to the loud first phrase can bury the terminal element while the label stays a dialect.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `snr_reference='ending'`, `snr_db_range=(6.0, 26.0)`.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.augment_ending_referenced_noise`.
# Training writes to `$STRNADI_WS/runs/augment_ending_referenced_noise/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'augment_ending_referenced_noise'

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
    snr_reference='ending',
    splice_probability=0.3,
    snr_db_range=(6.0, 26.0),
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
