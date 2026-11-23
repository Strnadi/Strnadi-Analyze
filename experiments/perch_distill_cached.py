from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, w_distill=0.5, teacher='cache',
    description='R7-clean + embedding distillation: the student\'s 128-d embedding is projected to Perch 2.0\'s 1536-d space and '
                'pulled towards the cached Perch embedding of the clean clip (cosine loss).',
    hypothesis='Perch\'s embedding space carries what 1.5 M recordings taught it; matching it regularises the small model. This '
               'is the BirdCLEF 2026 winning recipe and gives every BioAcu project head one shared embedding.',
    origin='research-session', status='proposed',
    result='Implemented in the research trainer (--w_distill) but never queued. Rows without a cached Perch embedding are '
           'masked out of the distillation term.',
    needs=CLEAN_NEEDS + '; emb/orig_perch.npz, and emb/rows_perch_emb.npy for the weak tables',
)
