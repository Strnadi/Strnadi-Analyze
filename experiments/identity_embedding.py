EXPERIMENT = dict(
    runner='identity', args='{name} {threads} 4000',
    description='id_v1: individual-identity embedding: StrnadNet + ArcFace over ring-coded Vizina males, scored by equal error '
                'rate and by synthetic two-bird diarisation.',
    hypothesis='A per-bird embedding lets a recording with n birds be split into each bird\'s songs (and gives per-bird dialects).',
    origin='research-session', status='queued',
    result='Not run. Audit note: male codes stop carrying over between years in 2024, so "different males" across that boundary '
           'may be the same birds.',
)
