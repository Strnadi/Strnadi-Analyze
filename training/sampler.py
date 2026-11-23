"""Sampling weights: class balance inside each data pool, then a fixed share of draws per pool.

A "pool" is one source table: the labelled clips ('strong') or one of the extra tables.
Without pools, the 2,000 "Unfinished" clips or one heavily sampled site (Vizina and JCU are
BhBl-only) would dominate, and synthetic counterfactual songs would swamp the 79 real BlBh clips
(independent-review finding V5).
"""

import numpy as np


def main_label(row):
    """The class a row is balanced by: its hard label, or the first member of its candidate set."""
    if row.label == 'soft':
        return row['hard']
    return row.label.split('|')[0]


def pool_sampling_weights(table, extra_shares, class_balance_power=0.5):
    """Compute one sampling weight per training row.

    Args:
        table: training rows with the columns `label` and `pool` (optionally `hard`, `sw`).
        extra_shares: {pool name: share of all draws}. Extra pools not listed share 50 % equally.
            The labelled pool 'strong' gets what is left (at least 5 %).
        class_balance_power: inside a pool, a row's weight is (count of its class) ** -power.

    Returns:
        (weights, realised_shares): weights sum to 1; realised_shares is {pool: share} for logging.
    """
    labels = table.apply(main_label, axis=1)

    if 'sw' in table:
        row_weights = table['sw'].fillna(1.0).values
    else:
        row_weights = np.ones(len(table))

    pools = list(table.pool.unique())
    extra_pools = [pool for pool in pools if pool != 'strong']
    default_share = 0.5 / len(extra_pools) if extra_pools else 0.0

    shares = {pool: extra_shares.get(pool, default_share) for pool in extra_pools}
    shares['strong'] = max(0.05, 1 - sum(shares.values()))

    weights = np.zeros(len(table))
    for pool in pools:
        in_pool = (table.pool == pool).values
        class_counts = labels[in_pool].value_counts()
        balance = labels[in_pool].map(lambda name: class_counts[name] ** (-class_balance_power))
        pool_weights = balance.values * row_weights[in_pool]
        weights[in_pool] = pool_weights / pool_weights.sum() * shares[pool]

    realised_shares = {}
    for pool in pools:
        realised_shares[pool] = round(float(weights[(table.pool == pool).values].sum()), 3)
    return weights, realised_shares
