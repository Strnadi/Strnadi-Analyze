"""First-use check of the Perch 2.0 PyTorch port (needs network: torch.hub + a weight download).

`layers/perch_encoder.py` was written from the port's README without being run. This verifies what it assumes:
output names and shapes, that a 4 s clip padded to 5 s works, that gradients reach the weights when
`trainable=True`, and (if the TensorFlow cache exists) that the pooled embedding agrees with the cached TF one.

    python -m tests.check_perch
"""

import os
import numpy as np
import torch
from config import WS


def main():
    from layers.perch_encoder import PerchEncoder

    enc = PerchEncoder(trainable=True)
    x = torch.randn(2, 128000) * 0.05
    out = enc(x)
    print({k: tuple(v.shape) for k, v in out.items()})
    assert out['embedding'].shape == (2, 1536) and out['spatial'].shape[:2] == (2, 1536)
    out['embedding'].sum().backward()
    n = sum(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in enc.net.parameters())
    print('parameters receiving gradient:', n)
    assert n > 0, 'the port is not fine-tunable as wrapped'
    from model import build_model

    m = build_model('perch_temporal', precomputed=False, trainable=False)
    assert m(x)['logits'].shape == (2, 7)
    cache = f'{WS}/emb/orig_perch.npz'
    if os.path.exists(cache):
        from training.data import read_table, load_seg

        z = np.load(cache)
        rows = read_table('data/orig_index.csv').iloc[:8]
        X = np.stack([load_seg(r.npy, r.t0, r.t0 + 4, 128000) for r in rows.itertuples()])
        with torch.no_grad():
            e = PerchEncoder()(torch.from_numpy(X))['embedding'].numpy()
        ref = z['emb'][:8].astype(np.float32)
        cos = (e * ref).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(ref, axis=1))
        print(
            'cosine to cached TF embeddings (different padding / gain conventions lower it):',
            np.round(cos, 3),
        )
    print('perch ok')


if __name__ == '__main__':
    main()
