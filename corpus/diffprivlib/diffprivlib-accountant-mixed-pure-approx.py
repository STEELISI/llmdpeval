import warnings

import numpy as np
from diffprivlib import mechanisms, tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_mean_and_gaussian_sum(df, mean_column, sum_column, bounds, epsilon, delta, mean_share=0.5):
    """Pure-DP mean (tools.mean) plus an (epsilon, delta)-DP Gaussian sum, both on one accountant.

    bounds=(lower, upper) are public for both columns. Returns ((mean, sum), (epsilon_spent, delta_spent)).
    """
    lower, upper = bounds
    acc = BudgetAccountant(epsilon=epsilon, delta=delta)
    e1, e2 = mean_share * epsilon, (1 - mean_share) * epsilon
    private_mean = tools.mean(df[mean_column].to_numpy(dtype=float), epsilon=e1, bounds=bounds, accountant=acc)
    acc.check(e2, delta)  # fail before drawing noise if the plan does not fit
    true_sum = float(np.clip(df[sum_column].to_numpy(dtype=float), lower, upper).sum())
    # Replace-one neighbours: one clipped value moves the sum by at most upper - lower.
    mech = mechanisms.GaussianAnalytic(epsilon=e2, delta=delta, sensitivity=upper - lower)
    private_sum = mech.randomise(true_sum)
    acc.spend(e2, delta)  # mechanism objects never charge an accountant themselves
    return (float(private_mean), private_sum), tuple(acc.total())


def _test():
    import pandas as pd
    a = np.arange(300) % 20
    df = pd.DataFrame({"a": a, "b": 20 - a})
    (m, s), (epsilon_spent, delta_spent) = dp_mean_and_gaussian_sum(df, "a", "b", (0, 20), 1.0, 1e-6)
    assert 0 <= m <= 20 and delta_spent <= 1e-6 + 1e-12
    return {"epsilon": epsilon_spent}
