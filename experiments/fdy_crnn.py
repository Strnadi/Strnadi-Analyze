from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='fdy_crnn', model_kw=dict(head='factorized'),
    description='FDY-CRNN: frequency-dynamic convolutions + BiGRU, the standard strong sound-event-detection network.',
    hypothesis='Kernels that adapt along frequency fit absolute-pitch patterns better than one shared kernel plus a coordinate '
               'channel.',
    origin='review-fork', status='proposed', result='Not run.', needs=CLEAN_NEEDS,
)
