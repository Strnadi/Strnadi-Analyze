from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN,
    description='R7 with the two data defects found in the fork review fixed: the noise bank purged of Yellowhammer songs, and '
                'noise level set against the song ending instead of the loud first phrase.',
    hypothesis='The research noise bank contains Yellowhammer songs (the R2 model flags 36 of 400 sampled chunks; at least 5 of '
               'the 12 viewed are clear songs with terminal elements) and is mixed into half of all training clips, including '
               '"Unfinished" and "None" ones; and 0-20 dB SNR relative to the phrase can still bury the fainter terminal element. '
               'Both corrupt exactly the complete-vs-Unfinished decision, where almost all remaining error is.',
    origin='review-fork', status='proposed',
    result='Not run. This is the base recipe (`recipes.CLEAN`) for the other proposals. The "ending" reference is a partial '
           'fix (2-6 dB on real clips; it finds the first terminal element, not the faintest one), so the recipe also raises '
           'the SNR range from 0-20 to 6-26 dB.',
    needs=CLEAN_NEEDS,
)
