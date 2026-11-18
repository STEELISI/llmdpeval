import warnings

from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_mean_and_count(df, value_column, bounds, flag_column, flag_value, epsilon, shares=(0.5, 0.5)):
    """Two releases under ONE total budget: a DP mean of df[value_column] and a DP count of
    df[flag_column] == flag_value. shares is fixed before touching the data and sums to 1.

    One accountant holds the total; it raises BudgetError if the calls would exceed it.
    Returns ((private_mean, private_count), epsilon_spent).
    """
    assert abs(sum(shares) - 1) < 1e-9
    eps_mean, eps_count = (s * epsilon for s in shares)
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    values = df[value_column].to_numpy(dtype=float)
    private_mean = tools.mean(values, epsilon=eps_mean, bounds=bounds, accountant=acc)
    mask = (df[flag_column] == flag_value).to_numpy()
    private_count = tools.count_nonzero(mask, epsilon=eps_count, accountant=acc)
    # acc.remaining() gives what is left for one more query; len(acc) is the number of recorded spends.
    return (float(private_mean), int(private_count)), acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(100) % 40, "group": ["a", "b"] * 50})
    (m, c), epsilon_spent = dp_mean_and_count(df, "value", (0, 50), "group", "a", epsilon=1.0)
    assert 0 <= m <= 50
    return {"epsilon": epsilon_spent}
