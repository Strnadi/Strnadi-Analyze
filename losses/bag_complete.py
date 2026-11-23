"""Recording-level bag loss (multiple-instance learning, noisy-OR) for the factorised head:
"a recording filed as dialect D contains at least one complete song of D".

For a bag of K songs from one recording:

    P(bag) = 1 - prod_i (1 - p_i),    p_i = p(song) p(complete) q(D),    loss = -log P(bag).

Unlike `factorized_weak_loss` this pushes on p(complete): some song in the bag has to be called
complete, and the model chooses which. That is the completeness supervision the 45k weak songs
otherwise never give. The original research plan listed this constraint; it was never implemented.

Use only for sources where the premise holds (NCS recordings labelled by a person who heard the
dialect); not for site-level labels (Vizina / AudioMoth / JCU), where a whole recording can be
unfinished songs.

Returns (n_bags,).

Origin: research plan + fork review.
"""

import torch

# Positions of the 5 dialects among the 7 classes; the other two (4, 5) are None and Unfinished.
DIALECT_IDX = [0, 1, 2, 3, 6]


def bag_complete_nll(out, dialect, bag_size):
    """Negative log-probability that a bag holds at least one complete song of its dialect.

    Args:
        out: model output dict of the factorised head for B = n_bags * bag_size songs, the songs
            of one bag next to each other, with
            `log_song` (B,) log p(song),
            `log_complete` (B,) log p(complete | song),
            `log_dialect` (B, 5) log q(dialect | complete), in the order of `DIALECT_IDX`.
            Padded bag members can be masked by setting their rows' `log_song` to -inf upstream.
        dialect: (n_bags,) integer index of each bag's dialect among the 5 dialects (0 ... 4).
        bag_size: number of songs per bag, K.

    Returns:
        (n_bags,) loss per bag (not averaged).
    """
    n_dialects = len(DIALECT_IDX)

    # For every song, pick the log-probability of ITS BAG's dialect out of the 5 dialects.
    log_q_all_dialects = out['log_dialect'].view(-1, bag_size, n_dialects)  # (n_bags, K, 5)
    dialect_index = dialect.view(-1, 1, 1).expand(-1, bag_size, 1)  # (n_bags, K, 1)
    log_q_bag_dialect = log_q_all_dialects.gather(2, dialect_index).squeeze(2)  # (n_bags, K)

    # log p_i = log p(song) + log p(complete | song) + log q(D | complete): the probability that
    # song i is a complete song of dialect D.
    log_p_complete_song = (out['log_song'] + out['log_complete']).view(-1, bag_size)  # (n_bags, K)
    log_p_hit = log_p_complete_song + log_q_bag_dialect  # (n_bags, K)

    # log prod_i (1 - p_i): the probability that NO song in the bag is such a song. The clamp
    # keeps log(1 - p) finite when p rounds to 1.
    p_hit = log_p_hit.exp().clamp_max(1 - 1e-6)
    log_p_no_hit = torch.log1p(-p_hit).sum(1)  # (n_bags,)

    # loss = -log P(bag) = -log(1 - exp(log_p_no_hit)).
    # Computed with the stable "log1mexp" form: clamping exp(...) below 1 instead would cap the
    # loss near 13.8 with ZERO gradient exactly for the bags the model gets most wrong (every
    # p_i tiny), which are the bags this loss exists for.
    log_p_no_hit = log_p_no_hit.clamp_max(-1e-12)
    near_zero = log_p_no_hit > -0.6931

    # Each formula is only evaluated where it is accurate; elsewhere it gets a harmless stand-in
    # value (-1), because `torch.where` would otherwise pass NaN gradients from the unused branch.
    stand_in = torch.full_like(log_p_no_hit, -1.0)
    for_expm1 = torch.where(near_zero, log_p_no_hit, stand_in)
    for_log1p = torch.where(near_zero, stand_in, log_p_no_hit)
    log_p_bag = torch.where(
        near_zero,
        torch.log(-torch.expm1(for_expm1)),
        torch.log1p(-torch.exp(for_log1p)),
    )
    return -log_p_bag
