from itertools import combinations

import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_perturb_categories(df, column, categories, distance, epsilon):
    """Local-DP perturbation of a categorical column with ExponentialCategorical.

    categories: public list of labels; distance(a, b) > 0: public dissimilarity (larger = less likely swap).
    Returns (perturbed values, epsilon_spent). Each record is randomised once.
    """
    utility_list = [(str(a), str(b), float(distance(a, b))) for a, b in combinations(categories, 2)]
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = mechanisms.ExponentialCategorical(epsilon=epsilon, utility_list=utility_list)
    noisy = np.array([mech.randomise(str(v)) for v in df[column]])
    acc.spend(epsilon, 0)  # one randomise call per person
    return noisy, acc.total()[0]


def _test():
    import pandas as pd
    levels = ["low", "mid", "high"]
    df = pd.DataFrame({"level": [levels[i % 3] for i in range(300)]})
    rank = {v: i for i, v in enumerate(levels)}
    noisy, epsilon_spent = dp_perturb_categories(df, "level", levels, lambda a, b: abs(rank[a] - rank[b]), 1.0)
    assert set(noisy) <= set(levels)
    return {"epsilon": epsilon_spent}
