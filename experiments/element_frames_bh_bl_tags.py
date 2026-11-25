# %% [markdown]
# # element_frames_bh_bl_tags
#
# Per-frame element tags with the two whistles of BhBl / BlBh named directly: `Bh` and `Bl`.
#
# **Question.** In `element_frames_from_tones` the first whistle is always `B` and the second is
# `L` or `H` (lower / higher than the B before), so only the second tag needs context. Here both
# whistles carry the ornithologists' names: BhBl is `Bh Bl`, BlBh is `Bl Bh`. The output reads
# better and the two tags must agree with each other, but the first whistle's tag now depends on
# what follows it (the same sound is `B` in a song that stops there). Which learns better?
#
# **Differs from `element_frames_from_tones` in:** the table
# `data/element_spans_from_tones_bh_bl.csv` (the same spans, built with `--direct-whistle-names`).
#
# **Status:** queued (2026-11-25).
#
# Run from the repository root with `python -m experiments.element_frames_bh_bl_tags`.
# Training writes to `$STRNADI_WS/runs/element_frames_bh_bl_tags/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'element_frames_bh_bl_tags'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/element_spans_from_tones_bh_bl.csv': 0.15,
        'data/weak_v5.csv': 0.25,
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
    # auxiliary losses
    element_loss_weight=1.0,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', element_frames=True)


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
