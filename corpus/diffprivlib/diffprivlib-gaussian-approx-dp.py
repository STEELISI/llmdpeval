import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_sum_gaussian(df, column, bounds, epsilon, delta):
    """(epsilon, delta)-DP sum of df[column] with the analytic Gaussian mechanism. bounds=(lower, upper) are public;
    delta comes from the task and should be well below 1 / n.

    Returns (private_sum, epsilon_spent).
    """
    lower, upper = bounds
    true_sum = float(np.clip(df[column].to_numpy(dtype=float), lower, upper).sum())
    sensitivity = upper - lower  # replace-one: one clipped value changes by at most upper - lower (L1 = L2 for a scalar)
    # GaussianAnalytic accepts any epsilon > 0; classic mechanisms.Gaussian raises ValueError for epsilon > 1;
    # mechanisms.GaussianDiscrete is for integer values with an integer sensitivity.
    acc = BudgetAccountant(epsilon=epsilon, delta=delta)  # approximate DP needs a delta ceiling
    acc.check(epsilon, delta)  # before the release: refuse an overspend
    mech = mechanisms.GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    private_sum = mech.randomise(true_sum)
    acc.spend(epsilon, delta)
    return private_sum, acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(1000) % 30})
    private_sum, epsilon_spent = dp_sum_gaussian(df, "value", bounds=(0, 50), epsilon=1.0, delta=1e-5)
    assert np.isfinite(private_sum)
    return {"epsilon": epsilon_spent}
