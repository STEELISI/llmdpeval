import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_sum(df, column, bounds, epsilon):
    """DP sum of df[column]. bounds=(lower, upper) are public; values outside are clipped before summing.

    Float input gets truncated Laplace noise. Returns (private_sum, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    values = df[column].to_numpy(dtype=float)
    private_sum = tools.sum(values, epsilon=epsilon, bounds=bounds, accountant=acc)
    return float(private_sum), acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(100) % 7})
    private_sum, epsilon_spent = dp_sum(df, "value", bounds=(0, 10), epsilon=1.0)
    assert np.isfinite(private_sum)
    return {"epsilon": epsilon_spent}
