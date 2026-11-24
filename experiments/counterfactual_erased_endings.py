# %% [markdown]
# # counterfactual_erased_endings
#
# The reference plus complete songs whose terminal element was erased, as "Unfinished" examples.
#
# **Question.** The one decision that still fails is "is the terminal element there?". Exact
# supervision for it: take a complete song, find its terminal element with the model's own
# occlusion map, replace it by background, and the song is unfinished by construction
# (`processing/build_erased_endings.py`). The earlier counterfactuals (`cf_v4`) did this by hand
# for BhBl songs of one site only; this covers every dialect and every recording.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** the extra table
# `data/erased_endings.csv` with 7 % of the training draws.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.counterfactual_erased_endings`.
# Training writes to `$STRNADI_WS/runs/counterfactual_erased_endings/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'counterfactual_erased_endings'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/weak_v5.csv': 0.33,
        'data/erased_endings.csv': 0.07,
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
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
