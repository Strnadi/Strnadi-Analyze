from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='strnadnet_measure', model_kw={}, w_measure=0.5,
    description='Concept bottleneck: the network regresses the expert measurements of the ending (f1, f2, f3, df, bw, t2, element '
                'presence) and the class is predicted from those values only.',
    hypothesis='Forcing the model to "measure like an ornithologist" makes it use the right evidence and makes every decision '
               'explainable in the scientists\' own terms.',
    origin='review-fork', status='proposed',
    result='Not run. Research backlog item "physics-informed multitask", never built. Model, loss and trainer hook are in place; '
           'the targets table is NOT built yet (docs/STATUS.md). Check Table S1\'s frequency columns first: its timing column is '
           'a 6.000 s placeholder in 384 of 430 rows.',
    needs=CLEAN_NEEDS + '; rows with columns m_f1, m_f2, m_f3, m_df, m_bw (kHz), m_t2 (s), m_has_first, m_has_second '
                        '(from Diblikova et al. 2023 Table S1; blank where unknown)',
)
