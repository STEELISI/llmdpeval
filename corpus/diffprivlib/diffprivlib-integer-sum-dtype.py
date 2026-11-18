import warnings

from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_integer_sum(df, column, bounds, epsilon):
    """DP integer total of df[column]. bounds=(lower, upper) are public integers.

    dtype=int makes diffprivlib use GeometricTruncated (discrete) noise. Returns (private_total, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    values = df[column].to_numpy()
    private_total = tools.sum(values, epsilon=epsilon, bounds=bounds, dtype=int, accountant=acc)
    return int(private_total), acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"units": np.arange(200) % 6})
    total, epsilon_spent = dp_integer_sum(df, "units", (0, 10), epsilon=1.0)
    assert isinstance(total, int) and 0 <= total <= 2000
    return {"epsilon": epsilon_spent}
