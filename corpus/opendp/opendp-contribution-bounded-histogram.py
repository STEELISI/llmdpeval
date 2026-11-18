import opendp.prelude as dp

dp.enable_features("contrib")


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
    capped = cap_contributions(df, unit_column, item_column, max_items)
    # After capping, one unit is at most max_items rows. Replace-one unit = remove up to max_items rows
    # and add up to max_items: symmetric distance d_in = 2 * max_items. OpenDP prices that distance.
    d_in = 2 * max_items
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    hist = space >> dp.t.then_count_by_categories(categories=list(items), null_category=False)
    meas = dp.binary_search_chain(lambda s: hist >> dp.m.then_laplace(s), d_in=d_in, d_out=epsilon)
    noisy = meas(capped[item_column].astype(str).tolist())
    return dict(zip(items, noisy)), meas.map(d_in)


def _test():
    import pandas as pd
    import numpy as np
    units = np.repeat(np.arange(400), np.random.randint(1, 9, 400))
    df = pd.DataFrame({"unit": units, "item": np.random.choice(list("abcdefgh"), len(units))})
    counts, eps = dp_bounded_item_histogram(df, "unit", "item", list("abcdefg"), max_items=3, epsilon=1.0)
    assert list(counts) == list("abcdefg")
    assert cap_contributions(df, "unit", "item", 3).groupby("unit").size().max() <= 3
    return {"epsilon": float(eps)}
