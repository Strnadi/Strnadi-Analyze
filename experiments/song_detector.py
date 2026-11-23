# %% [markdown]
# # song_detector
#
# Song-activity supervision: the per-frame output of the classifier is trained on known song
# spans, so the same model can find songs in a recording (model/song_detector.py is the stand-
# alone 0.1 M-parameter version).
#
# **Question.** End to end, missed detections are the largest loss (song recall 0.875 with the
# energy segmenter), the BirdNET >= 0.3 filter biases every training pool toward loud typical
# songs, and the phone app cannot ship BirdNET.
#
# **Status:** proposed (from review proposal). Not run. Frame targets come from `song_t0` /
# `song_t1` columns; the corrected table builders write them (processing/build_weak_set.py,
# processing/realign_orig.py -> song_s0/song_s1). Rows without them are ignored by the frame loss.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv; tables rebuilt with song spans
#
# Run from the repository root with `python -m experiments.song_detector`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/song_detector/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'song_detector'

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
    # weak-label and auxiliary losses
    frame_loss_weight=1.0,
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
