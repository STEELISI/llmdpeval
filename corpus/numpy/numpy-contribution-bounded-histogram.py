import numpy as np


def cap_contributions(df, unit_column, item_column, max_items):
    """Keep each unit's distinct items, then at most max_items of them, chosen at random within the unit.
    The rule looks only at the unit's own rows, so it is a per-unit transformation (no data-wide choice)."""
    distinct = df[[unit_column, item_column]].drop_duplicates()
    shuffled = distinct.sample(frac=1.0)                     # random order; no seed
    return shuffled.groupby(unit_column, sort=False).head(max_items)


def dp_bounded_item_histogram(df, unit_column, item_column, items, max_items, epsilon):
    """Count, for each PUBLIC item, how many units list it, when one unit (person, report) can list many
    items. Each unit keeps at most max_items distinct items (max_items PUBLIC, from a rule fixed in advance).
    Replace-one unit (bounded DP: the number of units is public).

    Returns ({item: noisy count}, epsilon_spent).
    """
    rng = np.random.default_rng()
    capped = cap_contributions(df, unit_column, item_column, max_items)
    idx = {it: i for i, it in enumerate(items)}
    codes = capped[item_column].map(idx).dropna().astype(int).to_numpy()   # items outside the list dropped
    counts = np.bincount(codes, minlength=len(items)).astype(float)
    # One unit adds 1 to at most max_items bins. Replace-one = remove one unit (-max_items in L1)
    # and add another (+max_items), so the L1 sensitivity of the count vector is 2 * max_items.
    sens = 2.0 * max_items
    scale = sens / epsilon
    noisy = counts + rng.laplace(0.0, scale, len(items))
    epsilon_spent = sens / scale
    return dict(zip(items, noisy.tolist())), epsilon_spent


def _test():
    rng = np.random.default_rng()
    import pandas as pd
    units = np.repeat(np.arange(400), rng.integers(1, 9, 400))
    df = pd.DataFrame({"unit": units, "item": rng.choice(list("abcdefgh"), len(units))})
    counts, eps = dp_bounded_item_histogram(df, "unit", "item", list("abcdefg"), max_items=3, epsilon=1.0)
    assert list(counts) == list("abcdefg")
    assert cap_contributions(df, "unit", "item", 3).groupby("unit").size().max() <= 3
    return {"epsilon": float(eps)}
