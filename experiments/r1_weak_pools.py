from .recipes import RESEARCH_PROTOCOL, WEAK_V3

EXPERIMENT = dict(
    RESEARCH_PROTOCOL, **WEAK_V3, model='strnadnet',
    description='R1: strong clips + 47.8k weak songs as candidate sets {file dialect, Unfinished, None} + counterfactual songs '
                '+ Vizina annotations, real AudioMoth noise, null splices. Flat 7-way head.',
    hypothesis='Does the big weakly labelled pool fix the citizen-recording errors (complete songs called Unfinished)?',
    origin='research-session', status='done',
    result='5 folds: macro-F1 0.754; vs base_v3 +0.009 [-0.009, +0.026], not significant. The weak data only pays off with the '
           'factorised head (R2).',
)
