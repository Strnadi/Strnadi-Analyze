# %% [markdown]
# # detector_band_tcn_scan_labels
#
# Song detector: a 1-D network over 40 coarse bands (BandTCN).
#
# **Question.** For live feedback on a phone a detector should run all the time and wake the
# classifier only when a Yellowhammer sings. How small can a network be and still find the songs?
# This one has about 8 thousand parameters. It is trained on "song / no song" labels that come from people and from
# BirdNET, never from our own models (`processing/build_detector_tables.py`); in those tables
# `Unfinished` stands for "song". `evaluation/detector_gate.py` then measures what matters: on
# whole recordings, how many annotated songs survive when the classifier only looks where the
# detector fires, and how much classifier work that saves.
#
# **This variant** also trains on the classifier's own scan: the 86.8 thousand songs it found
# and windows in which it hears nothing (`processing/build_detector_scan_table.py`), half of all
# draws. Compare with `detector_band_tcn`, which uses labels from people and BirdNET only.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** the detector tables, a plain
# classifier head, and the network.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.detector_band_tcn_scan_labels`.
# Training writes to `$STRNADI_WS/runs/detector_band_tcn_scan_labels/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.band_tcn import BandTCN
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'detector_band_tcn_scan_labels'

settings = TrainSettings(
    # data
    labelled_table='data/detector_clips.csv',
    extra_tables={'data/detector_extra.csv': 0.3, 'data/detector_scan.csv': 0.5},
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
    return BandTCN()


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
