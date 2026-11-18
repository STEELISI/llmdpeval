from itertools import combinations

import numpy as np


def dp_top_k_set_joint(df, column, categories, k, epsilon):
    """Top-k SET of public categories by count with ONE exponential-mechanism draw over all k-subsets.
    categories: PUBLIC list of every possible value, declared before looking at the data (codebook,
    schema), in a fixed order; never df[column].unique() (which values occur is itself private).

    Utility of a subset = sum of its counts; one row moves it by at most 1 under add/remove and under
    replace-one, so delta_u = 1. Not assumed monotone: weights exp(eps * u / (2 * delta_u))
    (McSherry & Talwar 2007; Dwork & Roth 2014, Thm 3.10). One draw costs epsilon, whatever k is.
    Returns (tuple of k categories in public-list order, epsilon_spent).
    """
    rng = np.random.default_rng()
    counts = df[column].value_counts()
    per_cat = np.array([float(counts.get(c, 0)) for c in categories])
    subsets = list(combinations(range(len(categories)), k))     # C(K, k) candidates: keep K small
    u = np.array([per_cat[list(s)].sum() for s in subsets])
    delta_u = 1.0
    coef = epsilon / (2 * delta_u)                              # weight exp(coef * u)
    logits = coef * u
    p = np.exp(logits - logits.max())                           # subtract max: avoids overflow
    pick = subsets[rng.choice(len(subsets), p=p / p.sum())]
    epsilon_spent = 2 * delta_u * coef                          # EM with exp(coef*u) is (2*delta_u*coef)-DP
    assert epsilon_spent <= epsilon + 1e-9
    return tuple(categories[i] for i in pick), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 60 + ["b"] * 40 + ["c"] * 25 + ["d"] * 1 + ["e"] * 10})
    top, epsilon_spent = dp_top_k_set_joint(df, "group", ["a", "b", "c", "d", "e"], k=3, epsilon=1.0)
    assert len(set(top)) == 3
    return {"epsilon": epsilon_spent}
