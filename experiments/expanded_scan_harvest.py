# %% [markdown]
# # expanded_scan_harvest
#
# The expanded training set plus the songs that only the model's own scan found.
#
# **Question.** The old detection chain (energy segmenter, BirdNET score >= 0.3) captured 81 % of
# the songs where every song is annotated; the model's scan with attention voting finds 99.5 %.
# `processing/harvest_songs_by_scan.py` scanned all continuous recordings, and the songs the old
# chain had missed (mostly faint and distant ones) were machine-labelled with the same three
# checks. Does adding them help? Compare with `expanded_machine_labels`, which has the same share
# of labelled clips.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** the extra tables
# `data/songs_machine_labelled.csv` (15 % of the draws) and
# `data/songs_scan_machine_labelled.csv` (10 %).
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.expanded_scan_harvest`.
# Training writes to `$STRNADI_WS/runs/expanded_scan_harvest/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'expanded_scan_harvest'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/songs_machine_labelled.csv': 0.15,
        'data/songs_scan_machine_labelled.csv': 0.10,
        'data/weak_v5.csv': 0.20,
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
