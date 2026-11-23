from .recipes import R2

EXPERIMENT = dict(
    R2, extra=R2['extra'] + ['data/jcu_train.csv'],
    pool_share={'weak_v3': 0.30, 'cf_v3': 0.08, 'vizann_v1': 0.07, 'jcu_train': 0.08},
    description='R6: R2 + 38k weak songs {BhBl, Unfinished, None} from 28 JCU AudioMoth recorders (passive, far-field).',
    hypothesis='Passive-recorder data makes the model robust in that domain.',
    origin='research-session', status='running',
    result='Fold 0 only: macro-F1 0.775 (R2 fold 0: 0.799). Review warnings: (1) every passive-recorder training song is '
           'constrained to BhBl, so "passive acoustics => BhBl" is a learnable shortcut, and the JCU "wrong-dialect rate" metric '
           'rewards it; this is the main risk for the Italian data. (2) JCU test recorders sit 120-170 m from training recorders. '
           'Check with evaluation/dialect_prior_shift.py before trusting it.',
)
