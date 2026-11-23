from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, inner_val=0.0, select='last', epochs=12,
    description='Fixed 12-epoch schedule, EMA weights, last epoch; no inner validation set.',
    hypothesis='Checkpoint selection on a ~380-clip inner set is noise (+-0.02 per epoch, as large as the effects being '
               'measured): over 18 finished folds last-epoch scored 0.760 vs 0.757 for the argmax rule. Dropping it returns 15 % '
               'of the training recordings (the set has 79 BlBh clips), and held-out F1 is flat from epoch 4-8, so 25 epochs are '
               'about twice as long as needed.',
    origin='review-base', status='proposed',
    result='Not run. The outer-fold curve is still logged every epoch, so overfitting stays visible; it is just not used to pick '
           'a checkpoint.',
    needs=CLEAN_NEEDS,
)
