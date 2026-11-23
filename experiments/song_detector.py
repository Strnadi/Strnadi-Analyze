from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='strnadnet', model_kw=dict(head='factorized'), w_frames=1.0,
    description='Song-activity supervision: the per-frame output of the classifier is trained on known song spans, so the same '
                'model can find songs in a recording (model/song_detector.py is the stand-alone 0.1 M-parameter version).',
    hypothesis='End to end, missed detections are the largest loss (song recall 0.875 with the energy segmenter), the BirdNET >= 0.3 '
               'filter biases every training pool toward loud typical songs, and the phone app cannot ship BirdNET.',
    origin='review-base', status='proposed',
    result='Not run. Frame targets come from `song_t0` / `song_t1` columns; the corrected table builders write them '
           '(processing/build_weak_set.py, processing/realign_orig.py -> song_s0/song_s1). Rows without them are ignored by the '
           'frame loss.',
    needs=CLEAN_NEEDS + '; tables rebuilt with song spans',
)
