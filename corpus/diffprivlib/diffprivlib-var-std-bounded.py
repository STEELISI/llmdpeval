import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_spread(df, column, bounds, epsilon, kind="std"):
    """DP variance (kind="var") or standard deviation (kind="std") of df[column], population form (ddof=0).

    bounds=(lower, upper) are public. std is the square root of one DP variance, so it costs the same epsilon.
    Returns (private_value, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    values = df[column].to_numpy(dtype=float)
    release = tools.std if kind == "std" else tools.var
    private_value = release(values, epsilon=epsilon, bounds=bounds, accountant=acc)
    return float(private_value), acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(200) % 50})
    private_std, epsilon_spent = dp_spread(df, "value", bounds=(0, 50), epsilon=1.0)
    assert private_std >= 0
    return {"epsilon": epsilon_spent}
