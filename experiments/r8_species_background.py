from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, species='data/species_bank.npy', p_species=0.3,
    description='R8: other-species calls mixed into the background of 30 % of training songs, 5-20 dB below the song ending.',
    hypothesis='Errors lean on context outside the song (59 % of occlusion importance vs 48 % when correct), driven by other '
               'birds and second songs; teaching the model to ignore them removes those errors.',
    origin='research-session', status='proposed',
    result='Not run (research backlog "R8 candidate" after the occlusion analysis).',
    needs=CLEAN_NEEDS + '; processing/build_species_bank.py (-> data/species_bank.npy from hardneg_v1.csv)',
)
