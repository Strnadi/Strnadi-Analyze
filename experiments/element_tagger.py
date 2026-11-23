from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='element_tagger', model_kw={}, w_elements=1.0, extra=CLEAN['extra'] + ['data/element_frames_v1.csv'],
    pool_share={**CLEAN['pool_share'], 'element_frames_v1': 0.1},
    description='Element-sequence model: every 20 ms frame is tagged background / phrase / B / C / D / E / X; the dialect is read '
                'off the terminal elements. Trained jointly with the clip labels.',
    hypothesis='"Unfinished" becomes a sequence without an ending, an unknown dialect becomes an unseen sequence (Italian data), '
               'the output shows where the dialect sits, and the same formulation serves any species on the BioAcu platform.',
    origin='review-fork', status='proposed',
    result='Not run. Frame supervision exists only for Vizina (component boxes) and the counterfactual surgery positions. The '
           'model, loss and trainer hook are in place; the table builder is NOT written yet: it has to map the Vizina phrase '
           'types A-F / component names onto phrase / B / C / D / E / X, which needs the annotators\' key (docs/STATUS.md).',
    needs=CLEAN_NEEDS + '; data/element_frames_v1.csv with an `elements` column: JSON [[name, start_s, end_s], ...] per row',
)
