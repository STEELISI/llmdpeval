import warnings

from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_iqr(df, column, bounds, epsilon, low=0.25, high=0.75):
    """DP interquartile range (or any quantile spread). bounds=(lower, upper) are public.

    One tools.quantile call with a list: each quantile gets epsilon/2 internally, total epsilon.
    Returns ((q_low, q_high, spread), epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    q = tools.quantile(df[column].to_numpy(dtype=float), [low, high], epsilon=epsilon, bounds=bounds,
                       accountant=acc)
    q_low, q_high = sorted(map(float, q))  # independent noise may swap them; sorting is post-processing
    return (q_low, q_high, q_high - q_low), acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(400) % 80})
    (q1, q3, spread), epsilon_spent = dp_iqr(df, "value", (0, 100), epsilon=1.0)
    assert 0 <= q1 <= q3 <= 100 and spread >= 0
    return {"epsilon": epsilon_spent}
