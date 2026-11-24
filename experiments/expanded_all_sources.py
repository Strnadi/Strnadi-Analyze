# %% [markdown]
# # expanded_all_sources
#
# Every machine-labelled table at once: archive and BirdNET songs, scan-harvested songs, and the
# passive-recorder (JCU AudioMoth) songs.
#
# **Question.** Does the largest training set help? The JCU songs are 28k machine-labelled songs
# from 28 passive recorders at ONE site with ONE dialect (BhBl or unfinished), so they get a small
# share of the draws: "passive recorder means BhBl" is a shortcut a model could learn.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** the extra tables
# `data/songs_machine_labelled.csv` (13 % of the draws), `data/songs_scan_machine_labelled.csv`
# (8 %) and `data/songs_jcu_machine_labelled.csv` (4 %).
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.expanded_all_sources`.
# Training writes to `$STRNADI_WS/runs/expanded_all_sources/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'expanded_all_sources'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/songs_machine_labelled.csv': 0.13,
        'data/songs_scan_machine_labelled.csv': 0.08,
        'data/songs_jcu_machine_labelled.csv': 0.04,
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
