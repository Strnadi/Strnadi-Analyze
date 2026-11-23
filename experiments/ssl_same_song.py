EXPERIMENT = dict(
    runner='ssl', args='--name {name} --threads {threads} --workers 3 --steps 12000 --bs 48 --p_other_song 0 --sources vizina,audiomoth_bhbl,birdnet_yh',
    description='ssl_v2: label-free pretraining of StrnadNet: same-song contrastive (two noisy / re-equalised views) + data2vec-style '
                'masked-frame prediction, on ~100k unlabelled songs (no NCS audio, no test males).',
    hypothesis='Label-free pretraining beats training from scratch once fine-tuned with the R2 recipe (--init).',
    origin='research-session', status='running',
    result='Not finished. Both reviews and the research session\'s own literature report rate self-supervised learning from '
           'scratch on CPU as a poor bet next to distilling Perch 2.0 (experiments/perch_distill_cached.py).',
)
