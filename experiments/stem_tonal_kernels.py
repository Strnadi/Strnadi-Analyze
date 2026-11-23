from .recipes import CLEAN, CLEAN_NEEDS, with_model_kw

EXPERIMENT = dict(
    with_model_kw(CLEAN, stem='tonal', stem_stride=(1, 2)),
    description='Stem ablation: long (1 x 9) and (1 x 15) time kernels beside the 3x3 branch; the stem no longer halves frequency.',
    hypothesis='Long horizontal kernels sum energy along thin tonal lines before any downsampling; the research stem (3x3, stride '
               '2 in both axes) sees 3 frames of a 0.3 s element and drops half the frequency resolution in the first layer.',
    origin='review-fork', status='proposed', result='Not run. About 2x the compute of the default stem (full frequency resolution '
                                                       'through stage 1).',
    needs=CLEAN_NEEDS,
)
