from .recipes import R7

EXPERIMENT = dict(
    model='perch_temporal', model_kw=dict(precomputed=True, head='factorized'), inputs='perch_spatial',
    strong=R7['strong'], extra=R7['extra'], pool_share=R7['pool_share'], lr=1e-3, epochs=20, epoch_size=16384, bs=64,
    description='The weak-label recipes (factorised head, pool shares, bag loss) searched on cached Perch embeddings, where a '
                'fold takes minutes instead of 2-4 CPU hours.',
    hypothesis='Recipe choices transfer from frozen-Perch features to the small model, so the search can run ~100x faster; the '
               'result is also a strong distillation teacher.',
    origin='review-fork', status='proposed', result='Not run.',
    needs='Perch embeddings for the weak / counterfactual / Vizina tables: processing/extract_foundation_emb.py <table> '
          '(-> emb/rows_perch_spatial.npy). The trainer refuses rows without a cached embedding.',
)
