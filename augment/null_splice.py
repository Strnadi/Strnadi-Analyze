"""Cut the clip at k random points and re-join it with the same 5 ms cross-fade the counterfactual
surgery uses.

The synthetic (swapped / truncated) songs contain such joins; if only they had them, "has a join"
would be a shortcut to their labels. Applying inaudible joins to every class removes it.

Origin: research session (independent-review fix V5).
"""

import numpy as np
from .fill import fill_to


def null_splice(y, k=None, fade=160):
    """Cut a clip into pieces and cross-fade them back together in the original order.

    Nothing is reordered, so the content stays the same; only the joins are new. Every join
    overlaps two pieces by `fade` samples, which makes the clip that much shorter, and `fill_to`
    pads the end back to the original length with noise. Uses the global `np.random` state.

    Args:
        y: waveform, 1-D float32 array.
        k: number of cuts; None (or 0) draws 1-4 at random.
        fade: length of the cross-fade in samples (160 = 5 ms at 32 kHz).

    Returns:
        A waveform with the same number of samples as `y`.
    """
    k = k or np.random.randint(1, 5)
    n = len(y)

    # Cut positions, somewhere between 20 % and 95 % of the clip.
    first_allowed = int(0.2 * n)
    last_allowed = int(0.95 * n)
    cut_points = sorted(np.random.randint(first_allowed, last_allowed, k))

    # Split the clip at the cut points into k + 1 consecutive pieces.
    pieces = []
    piece_start = 0
    for cut_point in cut_points:
        pieces.append(y[piece_start:cut_point])
        piece_start = cut_point
    pieces.append(y[piece_start:])

    # Join the pieces again. `fade_in` rises linearly from 0 to 1; the end of what is joined so
    # far fades out while the start of the next piece fades in.
    joined = pieces[0].copy()
    fade_in = np.linspace(0, 1, fade, dtype=np.float32)
    fade_out = 1 - fade_in
    for piece in pieces[1:]:
        both_long_enough = len(joined) >= fade and len(piece) >= fade
        if both_long_enough:
            joined[-fade:] = joined[-fade:] * fade_out + piece[:fade] * fade_in
            joined = np.concatenate([joined, piece[fade:]])
        else:
            # A piece shorter than the fade is appended without a cross-fade.
            joined = np.concatenate([joined, piece])

    return fill_to(joined, n)
