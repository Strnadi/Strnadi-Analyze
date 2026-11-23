"""Recording-level bag loss (multiple-instance learning, noisy-OR) for the factorised head:
"a recording filed as dialect D contains at least one complete song of D".

For a bag of K songs from one recording:  P(bag) = 1 - prod_i (1 - p_i),  p_i = p(song) p(complete) q(D),
loss = -log P(bag). Unlike `factorized_weak_loss` this pushes on p(complete): some song in the bag has to be
called complete, and the model chooses which. That is the completeness supervision the 45k weak songs
otherwise never give. The original research plan listed this constraint; it was never implemented.
Use only for sources where the premise holds (NCS recordings labelled by a person who heard the dialect);
not for site-level labels (Vizina / AudioMoth / JCU), where a whole recording can be unfinished songs.
Returns (n_bags,). Origin: research plan + fork review.
"""
import torch

DIALECT_IDX = [0, 1, 2, 3, 6]


def bag_complete_nll(out, dialect, bag_size):
    """out: model dict for B = n_bags * bag_size songs (bags contiguous); dialect: (n_bags,) index into the 5
    dialects; padded bag members can be masked by setting their rows' `log_song` to -inf upstream."""
    lq = out['log_dialect'].view(-1, bag_size, len(DIALECT_IDX))
    lp = (out['log_song'] + out['log_complete']).view(-1, bag_size) + lq.gather(2, dialect.view(-1, 1, 1).expand(-1, bag_size, 1)).squeeze(2)
    log_not = torch.log1p(-lp.exp().clamp_max(1 - 1e-6)).sum(1)          # log prod (1 - p_i)
    return -torch.log1p(-log_not.exp().clamp_max(1 - 1e-6))
