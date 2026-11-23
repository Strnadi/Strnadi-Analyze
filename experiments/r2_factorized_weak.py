from .recipes import R2

EXPERIMENT = dict(
    R2,
    description='R2: R1 + factorised head. Weak file-level labels train only the dialect factor, weighted by p(complete).',
    hypothesis='Weak labels help once they can no longer drag hard songs into "Unfinished" (the cheapest way to satisfy '
               '"BC or Unfinished or None").',
    origin='research-session', status='done',
    result='5 folds: macro-F1 0.774, acc 0.874; recall BhBl 0.55, BlBh 0.65, None 0.50, Unfinished 0.92, XB 0.95. '
           'vs base_v3 +0.029 [+0.013, +0.046] (paired recording bootstrap, one seed each); vs R1 +0.020 [+0.007, +0.034]. '
           'Expert bird-level BE/BlBh accuracy 0.939 (base 0.893, original NCS file labels 0.863). '
           'Trained with three since-fixed defects: black-patch counterfactual filler (cf_v3), clip-power SNR, 20 % of strong '
           'clips starting mid-song. Complete-vs-Unfinished AUC: BhBl 0.83, BlBh 0.89, BC 0.93, BE 0.95, XB 0.98.',
)
