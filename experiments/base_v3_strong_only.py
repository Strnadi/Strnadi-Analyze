from .recipes import RESEARCH_PROTOCOL

EXPERIMENT = dict(
    RESEARCH_PROTOCOL, model='strnadnet',
    description='Baseline: StrnadNet (0.9 M params) on the 3,744 human-labelled clips only, recording-grouped 5-fold CV.',
    hypothesis='Reference point every other recipe has to beat on the same folds.',
    origin='research-session', status='done',
    result='5 folds: macro-F1 0.745, acc 0.868; recall BC 0.82, BE 0.85, BhBl 0.50, BlBh 0.49, None 0.43, Unfinished 0.93, XB 0.93. '
           'Dialect accuracy given a complete song 0.974; BhBl<->BlBh confusions 1 of 223; 69 of 144 BhBl clips predicted Unfinished. '
           'Held-out Vizina males (aligned clips): BhBl recall 0.96 vs 0.68 for the production model. '
           'NCS bird-level accuracy 0.957 at coverage 0.89. ECE 0.020. Training loss reaches the label-smoothing floor by epoch ~12 '
           'while held-out F1 is flat from epoch 4-8: the labelled set is memorised.',
)
