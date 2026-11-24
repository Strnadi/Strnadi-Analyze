# %% [markdown]
# # golf_190k_half_resolution
#
# Parameter and speed golf: the 0.19 M network on the half-resolution spectrogram.
#
# **Question.** The smallest network and the cheapest input, combined: how much accuracy is left at roughly a
# tenth of the reference's compute? (Phone candidate.)
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** widths 8-16-32-64, `frontend=dict(n_bins=64, hop=640)`.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.golf_190k_half_resolution`.
# Training writes to `$STRNADI_WS/runs/golf_190k_half_resolution/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'golf_190k_half_resolution'

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
    return StrnadNet(
        widths=(8, 16, 32, 64),
        d=64,
        emb_dim=64,
        head='factorized',
        frontend=dict(n_bins=64, hop=640),
    )


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
