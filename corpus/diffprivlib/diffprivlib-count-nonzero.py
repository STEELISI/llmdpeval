import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_count_where(df, column, value, epsilon):
    """DP count of rows with df[column] == value. value comes from the task, not from inspecting the data.

    count_nonzero is a sum with bounds (0, 1) and integer dtype: geometric noise, integer output.
    Returns (private_count, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    mask = (df[column] == value).to_numpy()
    private_count = tools.count_nonzero(mask, epsilon=epsilon, accountant=acc)
    return int(private_count), acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    private_count, epsilon_spent = dp_count_where(df, "group", "a", epsilon=1.0)
    assert isinstance(private_count, int)
    return {"epsilon": epsilon_spent}
