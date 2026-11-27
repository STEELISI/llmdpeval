import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_mean(df, column, bounds, epsilon):
    """DP mean of df[column]. bounds=(lower, upper) must be public (codebook, legal range), never the data's min/max.

    Values are clipped to bounds; the output always lies inside them. Returns (private_mean, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    values = df[column].to_numpy(dtype=float)
    private_mean = tools.mean(values, epsilon=epsilon, bounds=bounds, accountant=acc)
    return float(private_mean), acc.total()[0]  # epsilon as recorded by diffprivlib's accountant


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(100) % 40 + 10})
    private_mean, epsilon_spent = dp_mean(df, "value", bounds=(0, 60), epsilon=1.0)
    assert 0 <= private_mean <= 60
    return {"epsilon": epsilon_spent}
