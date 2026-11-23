from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, bags=dict(table='data/weak_v3.csv', k=6, n_bags=4, sources=('ncs',)), w_bag=0.5,
    description='R7-clean + recording-level bag loss: of 6 songs drawn from an NCS recording filed as dialect D, at least one must '
                'be a complete song of D (noisy-OR over p(song) p(complete) q(D)).',
    hypothesis='With the factorised head the 45k weak songs teach nothing about completeness, the model\'s actual weakness. The '
               'bag constraint supplies that signal: some song in the recording has to be called complete.',
    origin='review-fork', status='proposed',
    result='Not run. The constraint was in the research session\'s first plan ("at least one song in the recording is D") and '
           'was never implemented. Not applied to site-level labels (Vizina / AudioMoth / JCU).',
    needs=CLEAN_NEEDS,
)
