from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, strong='data/orig_aligned_disputed.csv',
    description='Train the disputed dialect clips as candidate sets {label, Unfinished} instead of hard labels; evaluate on the '
                'original labels.',
    hypothesis='About 310 errors are shared by four independently trained recipes, and in 163 clips their average confidently '
               'contradicts the label (mostly "dialect" clips showing a phrase and ONE terminal element, which the project\'s own '
               'rule calls Unfinished). Training on them as hard dialect labels teaches the model to call incomplete songs a '
               'dialect. Until experts re-label them, a candidate set is the honest target.',
    origin='review-base', status='proposed',
    result='Not run. Sensitivity bound (circular, not a result): treating the confident disagreements as label errors moves '
           'macro-F1 from 0.78 to 0.85-0.88.',
    needs=CLEAN_NEEDS + '; processing/build_disputed.py (-> data/orig_aligned_disputed.csv with a `train_label` column)',
)
