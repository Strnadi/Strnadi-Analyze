# %% [markdown]
# # measurement_bottleneck
#
# Concept bottleneck: the class is decided from eight measurements of the song ending only.
#
# **Question.** Ornithologists tell the dialects apart by a few measurements of the ending: the
# pitch at the end of the first terminal element (f1), at the start and the end of the second
# (f2, f3), their differences (df, bw), the length of the second element (t2) and whether each
# element is there. A network forced to predict exactly these eight values, and to classify from
# them alone (`layers/measurement_head.py`), explains every decision in the scientists' own terms.
# How much accuracy does the bottleneck cost, and how close are its measurements?
#
# The targets are measured automatically inside the element spans found from label and
# spectrogram (`processing/build_measurements_from_tones.py` -> `data/clips_audited_measured.csv`,
# 3,232 of 3,597 clips); clips without spans train the class only.
#
# **Differs from `final_combined` in:** the head (8 measurements -> small MLP -> 7 classes) and
# `measurement_loss_weight=0.5`.
#
# **Status:** queued (2026-11-25), final label table. Held-out measurement error:
# `python -m evaluation.measurement_bottleneck measurement_bottleneck`.
#
# Run from the repository root with `python -m experiments.measurement_bottleneck`.
# Training writes to `$STRNADI_WS/runs/measurement_bottleneck/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet_measure import MeasurementStrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'measurement_bottleneck'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited_measured.csv',
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
    # auxiliary losses
    measurement_loss_weight=0.5,
)


# %% Model
def make_model():
    return MeasurementStrnadNet()


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
