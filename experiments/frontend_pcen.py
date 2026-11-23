from .recipes import CLEAN, CLEAN_NEEDS, with_model_kw

EXPERIMENT = dict(
    with_model_kw(CLEAN, frontend=dict(norm='pcen')),
    description='Front-end ablation: PCEN (adaptive gain control + root compression) instead of log + median.',
    hypothesis='PCEN is markedly more robust in noise and far field for bird audio (89.9 % vs 70.4 % for log-mel in one '
               'bird-activity study) and should keep faint terminal elements visible.',
    origin='research-session', status='proposed',
    result='Not run (in the research experiment matrix, recommended by its literature report, never queued).',
    needs=CLEAN_NEEDS,
)
