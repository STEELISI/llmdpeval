import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant


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
    idx = {it: i for i, it in enumerate(items)}
    codes = capped[item_column].map(idx).dropna().astype(int).to_numpy()   # items outside the list dropped
    acc = BudgetAccountant(epsilon=epsilon / (2 * max_items), delta=0)   # capped at what it records
    k = len(items)
    # diffprivlib calibrates a histogram for one row = one bin (sensitivity 1). Here one unit touches up to
    # max_items bins, and replace-one removes one unit and adds another: L1 sensitivity 2 * max_items.
    # So pass epsilon / (2 * max_items); the real guarantee is 2 * max_items times what the accountant records.
    noisy, _ = tools.histogram(codes, epsilon=epsilon / (2 * max_items), bins=k, range=(-0.5, k - 0.5),
                               accountant=acc)
    epsilon_spent = 2 * max_items * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return dict(zip(items, noisy.tolist())), epsilon_spent


def _test():
    import pandas as pd
    units = np.repeat(np.arange(400), np.random.randint(1, 9, 400))
    df = pd.DataFrame({"unit": units, "item": np.random.choice(list("abcdefgh"), len(units))})
    counts, eps = dp_bounded_item_histogram(df, "unit", "item", list("abcdefg"), max_items=3, epsilon=1.0)
    assert list(counts) == list("abcdefg")
    assert cap_contributions(df, "unit", "item", 3).groupby("unit").size().max() <= 3
    return {"epsilon": float(eps)}
