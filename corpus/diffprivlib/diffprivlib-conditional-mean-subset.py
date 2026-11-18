import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_subset_mean(df, value_column, mask, bounds, epsilon, sum_share=0.5):
    """DP mean of df[value_column] over rows where `mask` is True, as noisy sum / noisy count.

    bounds=(lower, upper) are public and must contain 0. The array length stays n (public), so the subset
    size is never treated as public. Returns (private_mean, epsilon_spent).
    """
    lower, upper = bounds
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    mask = np.asarray(mask, dtype=bool)
    masked = np.where(mask, np.clip(df[value_column].to_numpy(dtype=float), lower, upper), 0.0)
    dp_sum = tools.sum(masked, epsilon=sum_share * epsilon, bounds=bounds, accountant=acc)
    dp_count = tools.count_nonzero(mask, epsilon=(1 - sum_share) * epsilon, accountant=acc)
    private_mean = float(np.clip(dp_sum / max(dp_count, 1), lower, upper))  # post-processing
    return private_mean, acc.total()[0]


def _test():
    import pandas as pd
    a = np.arange(600)
    df = pd.DataFrame({"value": a % 50, "group": np.where(a % 3 == 0, "x", "y")})
    m, epsilon_spent = dp_subset_mean(df, "value", df["group"] == "x", (0, 50), epsilon=1.0)
    assert 0 <= m <= 50
    return {"epsilon": epsilon_spent}
