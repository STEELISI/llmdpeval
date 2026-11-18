import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.validation import check_bounds, clip_to_bounds


def dp_column_totals(df, columns, bounds, epsilon):
    """DP total of each column. bounds=(lower, upper): scalars or one public value per column.

    check_bounds normalises them like diffprivlib's tools; epsilon is split evenly over the columns.
    Returns (dict column -> private total, epsilon_spent).
    """
    d = len(columns)
    lower, upper = check_bounds(bounds, shape=d)  # two length-d float arrays; raises on bad input
    X = clip_to_bounds(df[list(columns)].to_numpy(dtype=float), (lower, upper))
    n = X.shape[0]  # public under bounded (replace-one) DP
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    out = {}
    for j, c in enumerate(columns):
        # Replace-one: one row changes column j's total by at most upper_j - lower_j.
        mech = mechanisms.LaplaceTruncated(epsilon=epsilon / d, sensitivity=upper[j] - lower[j],
                                           lower=n * lower[j], upper=n * upper[j])
        out[c] = mech.randomise(float(X[:, j].sum()))
    acc.spend(epsilon, 0)  # d releases of epsilon/d each
    return out, acc.total()[0]


def _test():
    import pandas as pd
    a = np.arange(200) % 10
    df = pd.DataFrame({"a": a, "b": a * 2.0})
    out, epsilon_spent = dp_column_totals(df, ["a", "b"], ([0, 0], [10, 20]), epsilon=1.0)
    assert 0 <= out["a"] <= 2000 and 0 <= out["b"] <= 4000
    return {"epsilon": epsilon_spent}
