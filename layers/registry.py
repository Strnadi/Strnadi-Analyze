"""Builders that turn a few config words into layer instances, so model files stay declarative."""

import torch.nn as nn


def build_frontend(
    bank='mel',
    norm='median',
    n_bins=128,
    fmin=1500.0,
    fmax=11000.0,
    n_fft=1024,
    hop=320,
    sr=32000,
    **norm_kw,
):
    """Waveform (B, N) -> spectrogram (B, C, n_bins, N / hop) front-end.

    bank: mel | linear
    norm: median | percentile | meanstd | pcen | magma | delta   (`norm_kw` goes to its constructor)
    n_fft: an int, or a list of ints for a multi-resolution stack (one channel group per window
    length).
    """
    from .db_delta_channels import DbDeltaChannels
    from .db_minmax_magma import DbMinMaxMagma
    from .linear_filterbank import LinearFilterbank
    from .log_meanstd_norm import LogMeanStdNorm
    from .log_median_norm import LogMedianNorm
    from .log_percentile_norm import LogPercentileNorm
    from .mel_filterbank import MelFilterbank
    from .pcen import PCEN
    from .spec_frontend import SpecFrontend

    bank_classes = {
        'mel': MelFilterbank,
        'linear': LinearFilterbank,
    }
    norm_classes = {
        'median': LogMedianNorm,
        'percentile': LogPercentileNorm,
        'meanstd': LogMeanStdNorm,
        'pcen': PCEN,
        'magma': DbMinMaxMagma,
        'delta': DbDeltaChannels,
    }

    def single_resolution(window_length):
        filterbank = bank_classes[bank](
            n_bins=n_bins, fmin=fmin, fmax=fmax, n_fft=window_length, sr=sr
        )
        normalisation = norm_classes[norm](**norm_kw)
        return SpecFrontend(filterbank, normalisation, window_length, hop)

    if isinstance(n_fft, (list, tuple)):
        if len(n_fft) == 1:
            return single_resolution(n_fft[0])
        from .multires_frontend import MultiResFrontend

        members = [single_resolution(window_length) for window_length in n_fft]
        return MultiResFrontend(members)
    return single_resolution(n_fft)


def build_temporal(kind, d, n_layers=3, drop=0.1):
    """Temporal module over a frame series, (B, D, T) -> (B, D, T).

    kind: conv | gru | attn | none
    """
    if kind == 'conv':
        from .temporal_dilated_block import TemporalDilatedBlock

        # the dilation doubles with every block: 1, 2, 4, ...
        blocks = []
        for i in range(n_layers):
            blocks.append(TemporalDilatedBlock(d, 5, 2**i, drop=drop))
        return nn.Sequential(*blocks)
    if kind == 'gru':
        from .temporal_bigru import TemporalBiGRU

        n_gru_layers = max(1, n_layers // 2)
        return TemporalBiGRU(d, n_gru_layers)
    if kind == 'attn':
        from .temporal_transformer import TemporalTransformer

        return TemporalTransformer(d, n_layers, drop=drop)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(kind)


def build_head(kind, d, n_classes, **kw):
    """Clip-embedding heads, (B, D) -> a dict with `logits` (B, n_classes).

    kind: linear | factorized | cosine | arcface | measurement | mlp
    `kw` goes to the head's constructor; it is ignored for linear and factorized.
    """
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
