# %% [markdown]
# # recording_mil
#
# Bird-level model: a small transformer + gated-attention pooling over all songs of a recording
# (each song given as its seven class log-probabilities), trained on the 3,333 file-level dialect labels as bag labels.
#
# **Question.** The scientific output is one dialect per bird. As bag labels the file labels are
# correct supervision, and a learned aggregator beats averaging per-song probabilities (bird-level
# accuracy is already 0.957 at 0.89 coverage by simple averaging).
#
# **Status:** queued (2026-11-25) on the song embeddings of `final_combined`.
#
# **Needs.** out-of-fold song embeddings of a finished run: evaluation/research_session/embed_songs.py <run> (->
# results/song_emb_<run>.npz)
#
# Run from the repository root with `python -m experiments.recording_mil`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/recording_mil/`.

# %% Imports
from model.recording_mil import RecordingMIL
from training.bird_level import fit_bird_level

# %% Settings
RUN = 'recording_mil'
SONG_RUN = 'final_combined'  # the run whose out-of-fold song embeddings are used


# %% Model
def make_model(embedding_dim):
    return RecordingMIL(d_in=embedding_dim)


# %% Train and evaluate (out of fold over recordings)
# Prints the bag model's bird-level accuracy next to the baseline it has to beat:
# the mean dialect probability over the songs the song model calls complete.
if __name__ == '__main__':
    fit_bird_level(RUN, make_model, song_run=SONG_RUN, epochs=30)
