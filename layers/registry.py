"""Builders that turn a few config words into layer instances, so model files stay declarative."""
import torch.nn as nn


def build_frontend(bank='mel', norm='median', n_bins=128, fmin=1500.0, fmax=11000.0, n_fft=1024, hop=320, sr=32000, **norm_kw):
    """bank: mel | linear      norm: median | percentile | meanstd | pcen | magma | delta
    n_fft: an int, or a list of ints for a multi-resolution stack (one channel group per window length)."""
    from .spec_frontend import SpecFrontend
    from .mel_filterbank import MelFilterbank
    from .linear_filterbank import LinearFilterbank
    from .log_median_norm import LogMedianNorm
    from .log_percentile_norm import LogPercentileNorm
    from .log_meanstd_norm import LogMeanStdNorm
    from .pcen import PCEN
    from .db_minmax_magma import DbMinMaxMagma
    from .db_delta_channels import DbDeltaChannels
    banks = {'mel': MelFilterbank, 'linear': LinearFilterbank}
    norms = {'median': LogMedianNorm, 'percentile': LogPercentileNorm, 'meanstd': LogMeanStdNorm, 'pcen': PCEN,
             'magma': DbMinMaxMagma, 'delta': DbDeltaChannels}

    def one(nf):
        return SpecFrontend(banks[bank](n_bins=n_bins, fmin=fmin, fmax=fmax, n_fft=nf, sr=sr), norms[norm](**norm_kw), nf, hop)

    if isinstance(n_fft, (list, tuple)):
        if len(n_fft) == 1:
            return one(n_fft[0])
        from .multires_frontend import MultiResFrontend
        return MultiResFrontend([one(nf) for nf in n_fft])
    return one(n_fft)


def build_temporal(kind, d, n_layers=3, drop=0.1):
    """(B, D, T) -> (B, D, T).   kind: conv | gru | attn | none"""
    if kind == 'conv':
        from .temporal_dilated_block import TemporalDilatedBlock
        return nn.Sequential(*[TemporalDilatedBlock(d, 5, 2 ** i, drop=drop) for i in range(n_layers)])
    if kind == 'gru':
        from .temporal_bigru import TemporalBiGRU
        return TemporalBiGRU(d, max(1, n_layers // 2))
    if kind == 'attn':
        from .temporal_transformer import TemporalTransformer
        return TemporalTransformer(d, n_layers, drop=drop)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(kind)


def build_head(kind, d, n_classes, **kw):
    """Clip-embedding heads; each returns a dict with `logits`.
    kind: linear | factorized | cosine | arcface | measurement | mlp"""
    if kind == 'linear':
        from .linear_head import LinearHead
        return LinearHead(d, n_classes)
    if kind == 'factorized':
        from .factorized_head import FactorizedHead
        return FactorizedHead(d, n_classes)
    if kind == 'cosine':
        from .subcenter_cosine_head import SubcenterCosineHead
        return SubcenterCosineHead(d, n_classes, **kw)
    if kind == 'arcface':
        from .arcface_head import ArcFaceHead
        return ArcFaceHead(d, n_classes, **kw)
    if kind == 'measurement':
        from .measurement_head import MeasurementHead
        return MeasurementHead(d, n_classes, **kw)
    if kind == 'mlp':
        from .mlp_head import MLPHead
        return MLPHead(d, n_classes, **kw)
    raise ValueError(kind)
