import warnings

from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_summary(df, column, bounds, epsilon, shares=(0.4, 0.3, 0.3)):
    """DP mean, std and median of df[column] under one total epsilon split by `shares` (sum to 1).

    bounds=(lower, upper) are public. Returns (dict of statistics, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)  # raises BudgetError if the shares overspend
    values = df[column].to_numpy(dtype=float)
    eps_mean, eps_std, eps_median = (s * epsilon for s in shares)
    stats = {
        "mean": float(tools.mean(values, epsilon=eps_mean, bounds=bounds, accountant=acc)),
        "std": float(tools.std(values, epsilon=eps_std, bounds=bounds, accountant=acc)),
        "median": float(tools.median(values, epsilon=eps_median, bounds=bounds, accountant=acc)),
    }
    return stats, acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(500) % 70 + 5})
    stats, epsilon_spent = dp_summary(df, "value", (0, 100), epsilon=1.0)
    assert set(stats) == {"mean", "std", "median"}
    return {"epsilon": epsilon_spent}
