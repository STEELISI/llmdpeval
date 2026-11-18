import warnings

from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_means_advanced(df, bounds, epsilon, delta, slack):
    """DP mean of every column in bounds (a dict {column: (lower, upper)} of public bounds), under ONE
    (epsilon, delta) ceiling with advanced composition. Only for tasks that allow a nonzero delta; 0 <= slack <= delta.

    The per-query share comes from the accountant itself (remaining(k) accounts for the slack).
    Replace-one (n public): tools.mean uses sensitivity (U - L)/n, so no halving.
    Returns ({column: private_mean}, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=delta, slack=slack)
    eps_each, delta_each = acc.remaining(len(bounds))
    result = {}
    for i, (column, col_bounds) in enumerate(bounds.items()):
        values = df[column].to_numpy(dtype=float)
        e = min(eps_each, acc.remaining(len(bounds) - i)[0])  # float rounding: never ask for more than is left
        result[column] = float(tools.mean(values, epsilon=e, bounds=col_bounds, accountant=acc))
    # acc.total() is the smaller of the plain sum and the advanced-composition bounds; its delta includes the slack.
    # Plain-sum total for comparison: BudgetAccountant().total(spent_budget=acc.spent_budget, slack=0)
    return result, acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    cols = [f"c{i}" for i in range(40)]
    df = pd.DataFrame({c: np.arange(200) % 30 - 15 for c in cols})
    for k in (2, 6, 9, 20, 40):   # k equal shares used to round above the ceiling for some k
        means, epsilon_spent = dp_means_advanced(df, {c: (-15, 15) for c in cols[:k]}, 1.0, 1e-5, 1e-6)
        assert len(means) == k and epsilon_spent <= 1 + 1e-9
    return {"epsilon": epsilon_spent}
