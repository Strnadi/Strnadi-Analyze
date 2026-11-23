from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='ending_zoom', model_kw={},
    description='Detect-then-zoom: a coarse network over the whole window plus a high-resolution (27 Hz, 5 ms) network on the '
                '1.5 s around the end of the song.',
    hypothesis='Given a complete song the dialect is right 97 % of the time; the errors are a missed faint ending. Resolution '
               'spent only where the ending is costs little and makes the 100-250 Hz criteria 4-9 bins wide.',
    origin='review-fork', status='proposed',
    result='Not run. The research session listed a "two-resolution zoom model" and never built it.',
    needs=CLEAN_NEEDS,
)
