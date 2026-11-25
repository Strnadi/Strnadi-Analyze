# %% [markdown]
# # recording_mil_fixed_embeddings
#
# The bird-level model on song embeddings, from a song model whose embedding space is fixed.
#
# **Question.** `recording_mil` has to work from the seven class probabilities of every song,
# because the embedding spaces of the five fold models are unrelated. The song model trained
# towards fixed class centres (`embedding_fixed_prototypes`) puts every fold into the same space.
# Do its 128-number embeddings tell the bird-level model more than the class probabilities do?
#
# **Differs from `recording_mil` in:** the song run and `song_features='embedding'`.
#
# **Status:** queued (2026-11-25).
#
# **Needs.** `python -m evaluation.research_session.embed_songs embedding_fixed_prototypes`
#
# Run from the repository root with `python -m experiments.recording_mil_fixed_embeddings`.
# Training writes to `$STRNADI_WS/runs/recording_mil_fixed_embeddings/`.

# %% Imports
from model.recording_mil import RecordingMIL
from training.bird_level import fit_bird_level

# %% Settings
RUN = 'recording_mil_fixed_embeddings'
SONG_RUN = 'embedding_fixed_prototypes'


# %% Model
def make_model(embedding_dim):
    return RecordingMIL(d_in=embedding_dim)


# %% Train and evaluate (out of fold over recordings)
if __name__ == '__main__':
    fit_bird_level(RUN, make_model, song_run=SONG_RUN, song_features='embedding', epochs=30)
