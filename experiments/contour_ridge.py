# %% [markdown]
# # contour_ridge
#
# Pitch-contour network on a ridge tracked finer than one frequency bin.
#
# **Question.** `contour_tcn` reads the pitch of the loudest tone from mel bins 60-70 Hz wide and
# reached 93.0 %. The published criteria for the dialects are differences of a few hundred Hz, so
# a contour measured to a few Hz (parabolic interpolation of the peak on a linear 39 Hz axis)
# could let a 0.27 M-parameter 1-D model apply them directly. Does the finer contour close the gap
# to the spectrogram network?
#
# **Differs from `final_combined` in:** the model (`ContourTCN(features='ridge')`, linear
# 192-bin spectrogram 2-9.5 kHz with percentile normalisation); training recipe unchanged.
#
# **Status:** queued (2026-11-25), final label table.
#
# Run from the repository root with `python -m experiments.contour_ridge`.
# Training writes to `$STRNADI_WS/runs/contour_ridge/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.contour_tcn import ContourTCN
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'contour_ridge'

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
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
    treble_cut_probability=0.25,
)


# %% Model
def make_model():
    return ContourTCN(
        features='ridge',
        head='factorized',
        frontend=dict(
            bank='linear', norm='percentile', n_bins=192, fmin=2000.0, fmax=9500.0, n_fft=2048
        ),
    )


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
