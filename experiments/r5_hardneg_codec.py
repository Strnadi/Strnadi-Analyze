from .recipes import R2

EXPERIMENT = dict(
    R2, extra=R2['extra'] + ['data/hardneg_v1.csv'],
    pool_share={'weak_v3': 0.33, 'cf_v3': 0.08, 'vizann_v1': 0.07, 'hardneg_v1': 0.05}, p_codec=0.15,
    description='R5: R2 + 6,865 other-species windows as "None" + phone-codec round-trips (AAC / Opus / MP3) on 15 % of clips.',
    hypothesis='Fewer false alarms on similar trills (Chiffchaff, Chaffinch, ...) and robustness to app audio compression.',
    origin='research-session', status='done',
    result='5 folds: macro-F1 0.761, acc 0.867 (computed from the out-of-fold predictions; not yet in the research log). '
           'None recall 0.55 (R2: 0.50), everything else within noise of R2 or slightly below.',
)
