from itertools import combinations

from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_top_k_set_joint(df, column, categories, k, epsilon, monotonic=False):
    """Top-k SET of public categories by count, in ONE exponential-mechanism draw over all k-subsets.
    categories: PUBLIC list of every possible value, declared before looking at the data (codebook,
    schema), in a fixed order; never df[column].unique() (which values occur is itself private).

    Candidates: every k-subset of the public list (enumerated from the list, not the data), so the
    candidate set is data-independent. Utility of a subset = sum of its counts. One row changes that
    sum by at most 1 under add/remove and under replace-one, so sensitivity = 1. monotonic=True is
    only valid under add/remove (all utilities move the same way). No k-fold budget split.
    Returns (tuple of k categories in public-list order, epsilon_spent).
    """
    counts = df[column].value_counts()
    per_cat = [float(counts.get(c, 0)) for c in categories]
    subsets = list(combinations(range(len(categories)), k))   # C(K, k) candidates: keep K small
    utility = [sum(per_cat[i] for i in s) for s in subsets]
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    mech = mechanisms.Exponential(epsilon=epsilon, sensitivity=1.0, utility=utility,
                                  monotonic=monotonic, candidates=list(range(len(subsets))))
    pick = subsets[mech.randomise()]
    acc.spend(epsilon, 0)                                     # mechanisms do not charge an accountant
    # The draw releases a SET; order inside it is the public list order, not the true ranking.
    return tuple(categories[i] for i in pick), acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 60 + ["b"] * 40 + ["c"] * 25 + ["d"] * 1 + ["e"] * 10})
    top, epsilon_spent = dp_top_k_set_joint(df, "group", ["a", "b", "c", "d", "e"], k=3, epsilon=1.0)
    assert len(set(top)) == 3
    return {"epsilon": epsilon_spent}
