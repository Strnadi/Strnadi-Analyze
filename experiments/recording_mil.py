EXPERIMENT = dict(
    runner='mil', args='--name {name} --songs r2_factorized_weak --threads {threads} --epochs 30',
    description='Bird-level model: a small transformer + gated-attention pooling over the embeddings of all songs in a recording, '
                'trained on the 3,333 file-level dialect labels as bag labels.',
    hypothesis='The scientific output is one dialect per bird. As bag labels the file labels are correct supervision, and a '
               'learned aggregator beats averaging per-song probabilities (bird-level accuracy is already 0.957 at 0.89 '
               'coverage by simple averaging).',
    origin='review-fork', status='proposed',
    result='Not run. Listed in the research backlog as "attention-MIL" / "bout-level set transformer", never built.',
    needs='out-of-fold song embeddings of a finished run: evaluation/embed_songs.py <run> (-> results/song_emb_<run>.npz)',
)
