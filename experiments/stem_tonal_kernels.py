# %% [markdown]
# # stem_tonal_kernels
#
# Stem ablation: long (1 x 9) and (1 x 15) time kernels beside the 3x3 branch; the stem no longer
# halves frequency.
#
# **Question.** Long horizontal kernels sum energy along thin tonal lines before any downsampling;
# the research stem (3x3, stride 2 in both axes) sees 3 frames of a 0.3 s element and drops half
# the frequency resolution in the first layer.
#
# **Status:** proposed (from review proposal). Not run. About 2x the compute of the default stem
# (full frequency resolution through stage 1).
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.stem_tonal_kernels`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/stem_tonal_kernels/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'stem_tonal_kernels'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
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
    return StrnadNet(head='factorized', stem='tonal', stem_stride=(1, 2))


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
