from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, strong='data/orig_aligned_observer_folds.csv',
    description='Robustness check of the protocol: folds grouped by observer (and duplicate-audio groups) instead of by recording.',
    hypothesis='Recording-grouped folds do not separate observers, sites or sessions: knowing the observer predicts the dialect '
               'with 0.81 accuracy (majority class 0.69), and 107 groups of byte-identical NCS recordings sit under different '
               'F-ids (81 span folds, some with conflicting labels). If scores drop under observer grouping, the model leans on '
               'recordist / device / site.',
    origin='review-base', status='proposed', result='Not run.',
    needs=CLEAN_NEEDS + '; processing/make_folds.py --group observer --dedupe (-> data/orig_aligned_observer_folds.csv)',
)
