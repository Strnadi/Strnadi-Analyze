# %% [markdown]
# # clean_noise_ending_snr
#
# R7 with the two data defects found in the fork review fixed: the noise bank purged of
# Yellowhammer songs, and noise level set against the song ending instead of the loud first
# phrase.
#
# **Question.** The research noise bank contains Yellowhammer songs (the R2 model flags 36 of 400
# sampled chunks; at least 5 of the 12 viewed are clear songs with terminal elements) and is mixed
# into half of all training clips, including "Unfinished" and "None" ones; and 0-20 dB SNR
# relative to the phrase can still bury the fainter terminal element. Both corrupt exactly the
# complete-vs-Unfinished decision, where almost all remaining error is.
#
# **Status:** proposed (from review proposal). Not run. This is the base recipe (`recipes.CLEAN`)
# for the other proposals. The "ending" reference is a partial fix (2-6 dB on real clips; it finds
# the first terminal element, not the faintest one), so the recipe also raises the SNR range from
# 0-20 to 6-26 dB.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.clean_noise_ending_snr`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/runs/clean_noise_ending_snr/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'clean_noise_ending_snr'

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
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
