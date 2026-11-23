from .recipes import R2

EXPERIMENT = dict(
    R2, unlabeled='data/unlabeled_v2.csv', w_u=1.0, workers=3,
    description='R4: R2 + FixMatch / FreeMatch consistency on 77.5k unlabelled songs (EMA teacher, adaptive thresholds).',
    hypothesis='Semi-supervised training inside the main loop is worth more than separate pretraining at this scale.',
    origin='research-session', status='running',
    result='In progress on 2026-11-23 (first attempt crashed at epoch 10 on an exhausted loader; fixed). Fold 0/1 outer F1 after 20 '
           'epochs: 0.80 / 0.76, i.e. inside the fold-to-fold spread of R2. Review caveat: the pseudo-labels come from the model '
           'itself, 78 % of unlabelled passive songs are predicted Unfinished, and BirdCLEF teams found weak self-teachers hurt.',
)
