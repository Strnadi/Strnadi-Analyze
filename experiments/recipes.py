"""Building blocks shared by experiment files (not experiments themselves)."""

# ---- exactly what the research session ran (R1-R6): original clip windows, clip-power SNR, black-patch counterfactuals
RESEARCH_PROTOCOL = dict(select='inner_argmax', eval_crop='start', snr_mode='clip')
WEAK_V3 = dict(extra=['data/weak_v3.csv', 'data/cf_v3.csv', 'data/vizann_v1.csv'],
               pool_share={'weak_v3': 0.35, 'cf_v3': 0.08, 'vizann_v1': 0.07},
               noise='data/noise_bank_audiomoth.npy', p_noise=0.5, p_splice=0.3, epoch_size=6144)
R2 = dict(**RESEARCH_PROTOCOL, **WEAK_V3, model='strnadnet', model_kw=dict(head='factorized'))

# ---- R7: R2 + the visual-QA fixes (song-aligned strong windows, spectrally matched counterfactual filler, band SNR)
R7 = dict(select='inner_argmax', eval_crop='start', snr_mode='band', strong='data/orig_aligned.csv',
          extra=['data/weak_v3.csv', 'data/cf_v4.csv', 'data/vizann_v1.csv'],
          pool_share={'weak_v3': 0.35, 'cf_v4': 0.08, 'vizann_v1': 0.07},
          noise='data/noise_bank_audiomoth.npy', p_noise=0.5, p_splice=0.3, epoch_size=6144,
          model='strnadnet', model_kw=dict(head='factorized'))

# ---- corrected base for everything proposed after the two reviews:
#      cleaned noise bank, SNR referenced to the song ending and 6 dB higher (the faintest terminal element is not
#      located by the energy rule, see augment/noise_mix.py), centred evaluation window, smoothed checkpoint rule
CLEAN = dict(R7, select='inner_smooth3', eval_crop='center', snr_mode='ending', snr=(6.0, 26.0), noise='data/noise_bank_clean.npy')
CLEAN_NEEDS = 'processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv; data/cf_v4.csv'


def with_model_kw(recipe, **kw):
    out = dict(recipe); out['model_kw'] = {**recipe.get('model_kw', {}), **kw}
    return out
