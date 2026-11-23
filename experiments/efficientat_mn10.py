# %% [markdown]
# # efficientat_mn10
#
# EfficientAT MobileNetV3 mn10 (4.9 M, AudioSet-pretrained, distilled from PaSST) + shared
# temporal head.
#
# **Question.** An audio-pretrained mobile CNN beats ImageNet initialisation; candidate phone
# model and cheap second teacher.
#
# **Status:** proposed (from review proposal). Not run. Recommended by the research session's own
# literature report; its sweep used ImageNet weights only. mn04_as (0.98 M) is the small variant.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv; EFFICIENTAT_DIR = clone of github.com/fschmid56/EfficientAT (model wrapper
# untested, see its docstring)
#
# Run from the repository root with `python -m experiments.efficientat_mn10`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/efficientat_mn10/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.efficientat_temporal import EfficientATTemporal
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'efficientat_mn10'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v3.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    eval_window='center',
    # protocol
    checkpoint_rule='inner_smooth3',
    # optimisation
    epochs=25,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.001,
    backbone_lr_factor=0.1,
    # waveform augmentation
    noise_probability=0.5,
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
)


# %% Model
def make_model():
    return EfficientATTemporal(pretrained_name='mn10_as', head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
