# %% [markdown]
# # expanded_machine_labels_round2
#
# Second turn of the labelling loop: the models trained WITH machine labels label the songs again.
#
# **Question.** Does repeating the loop (train -> label -> check -> train) keep improving the
# data? Teachers are now `expanded_machine_labels` (class head) and `element_sequence_ctc`
# (element sequence); the three checks and the per-fold labels are as in round 1.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** the extra table
# `data/songs_machine_labelled_round2.csv`, built by the first cell below, with a quarter of the
# training draws.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.expanded_machine_labels_round2`.
# Training writes to `$STRNADI_WS/runs/expanded_machine_labels_round2/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Label the songs again with the round-1 models (skipped when the table exists)
import os
import subprocess
import sys

from config import dry_run, workspace_path

TABLE = 'data/songs_machine_labelled_round2.csv'
if not dry_run() and not os.path.exists(workspace_path(TABLE)):
    teachers = ['expanded_machine_labels', 'element_sequence_ctc']
    command = [sys.executable, '-m', 'processing.machine_label_songs', '--out', TABLE, '--runs']
    subprocess.run(command + teachers, check=True)

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'expanded_machine_labels_round2'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/songs_machine_labelled_round2.csv': 0.25,
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
