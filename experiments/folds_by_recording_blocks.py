# %% [markdown]
# # folds_by_recording_blocks
#
# Cross-validation in which neighbouring archive recordings stay in the same fold.
#
# **Question.** Folds grouped by recording do not separate observers: one person often uploads
# several recordings of the same birds at the same place, and such recordings can land in
# different folds. Is the measured accuracy inflated by that? The table linking an archive
# recording to its observer covers too few recordings to group by observer directly, so
# neighbouring archive numbers stand in for "uploaded together": every block of 25 consecutive
# recording numbers is one group (`processing/build_block_folds.py`, 520 groups for 2,037
# recordings).
#
# **Differs from `ablation_clip_labels_only` in:** the folds only
# (`data/clips_audited_block_folds.csv`). Clip labels alone are used so that no extra table can
# carry a recording across the new fold borders.
#
# **Status:** queued (2026-11-25). Compare with `ablation_clip_labels_only` (96.0 %, macro-F1
# 0.909 on recording-grouped folds).
#
# Run from the repository root with `python -m experiments.folds_by_recording_blocks`.
# Training writes to `$STRNADI_WS/runs/folds_by_recording_blocks/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'folds_by_recording_blocks'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited_block_folds.csv',
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
    return StrnadNet()


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
