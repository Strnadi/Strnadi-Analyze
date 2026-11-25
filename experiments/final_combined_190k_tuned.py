# %% [markdown]
# # final_combined_190k_tuned
#
# The smallest full-resolution network with the combined recipe and the search winner's settings.
#
# **Question.** With the reference optimisation settings the 0.19 M network is 0.15 points behind
# the 0.9 M one on the combined recipe, a difference inside the seed spread. The tuned settings
# raised the 0.9 M network from 116-120 errors to 110. Does the small network gain as much, and
# does it then still match the large one?
#
# **Differs from `final_combined_tuned` in:** the network (stage widths 8, 16, 32, 64, temporal
# width 64 and a 64-number embedding, as in `final_combined_190k`).
#
# **Status:** queued (2026-11-25), three seeds.
#
# Run from the repository root with `python -m experiments.final_combined_190k_tuned`.
# Training writes to `$STRNADI_WS/runs/final_combined_190k_tuned/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'final_combined_190k_tuned'

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
    epochs=40,
    samples_per_epoch=6144,
    batch_size=96,
    learning_rate=0.005,
    weight_decay=0.1,
    ema_decay=0.995,
    early_stopping_patience=8,
    # waveform augmentation
    noise_probability=0.5,
    snr_reference='ending',
    snr_db_range=(6.0, 26.0),
    speed_jitter=0.04,
    splice_probability=0.3,
    treble_cut_probability=0.25,
)


# %% Model
def make_model():
    return StrnadNet(widths=(8, 16, 32, 64), d=64, emb_dim=64, head='factorized', drop=0.3)


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
