EXPERIMENT = dict(
    runner='ssl', args='--name {name} --threads {threads} --workers 3 --steps 12000 --bs 48 --p_other_song 0 --w_supcon 0 --w_mask 0 '
                       '--w_hubert 1 --codebook vq/cb_logmel_k100.npz --sources vizina,audiomoth_bhbl,birdnet_yh',
    description='ssl_hub: HuBERT-style masked prediction of k-means "acoustic units" (the objective behind AVES); doubles as a '
                'tokeniser of song elements (evaluation/vq_tokens.py).',
    hypothesis='Learned units make song endings clusterable into dialects without labels.',
    origin='research-session', status='queued',
    result='Not run. Stage-0 units (k-means on log-mel frames) carry no dialect information (ARI ~0); on supervised features '
           'kNN dialect-F1 0.30 vs 0.22 chance. Bag-of-units discards element order; a sequence model over units is needed.',
)
