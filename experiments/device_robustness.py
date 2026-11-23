from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, mixstyle_p=0.4, p_codec=0.15, p_eq=0.3,
    description='Device robustness: Freq-MixStyle + phone-codec round-trips + random device equalisation.',
    hypothesis='Phone microphones and codecs are the deployment domain and the one evaluated least; Freq-MixStyle + impulse-'
               'response augmentation took unseen-device accuracy from 48 % to 59 % in the cited DCASE study.',
    origin='research-session', status='proposed',
    result='Not run (research backlog item 9). Needs the app-database export to measure properly.',
    needs=CLEAN_NEEDS,
)
