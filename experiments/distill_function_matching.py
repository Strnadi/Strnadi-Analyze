from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, teacher='online', teacher_run='perch_finetune_teacher', w_distill=0.5, w_kd=1.0, epochs=60, win=5.0,
    unlabeled='data/unlabeled_v2.csv', w_u=0.0,
    description='The final small model: StrnadNet trained to match a fine-tuned teacher ensemble (soft labels + embeddings) on '
                'the SAME augmented views, over a long schedule.',
    hypothesis='"Function matching" (Beyer et al. 2022): consistent teaching on identical views with long schedules is what '
               'makes distillation work; EfficientAT gained +5.3 mAP this way.',
    origin='review-base', status='proposed',
    result='Not run. Extend to the ~100k unlabelled songs by pseudo-labelling them with the teacher '
           '(evaluation/pseudo_label.py) and adding the table to `extra`: pseudo-labels only from a strong teacher.',
    needs=CLEAN_NEEDS + '; a finished teacher run (experiments/perch_finetune_teacher.py)',
)
