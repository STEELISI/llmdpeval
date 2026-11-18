import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_sum_laplace(df, column, bounds, epsilon):
    """DP sum of df[column] with the Laplace mechanism used directly. bounds=(lower, upper) are public.

    Returns (private_sum, epsilon_spent).
    """
    lower, upper = bounds
    true_sum = float(np.clip(df[column].to_numpy(dtype=float), lower, upper).sum())
    # Replace-one neighbours: one clipped value moves from lower to upper, so the sum moves by at most upper - lower.
    # (Add/remove neighbours would give max(|lower|, |upper|) instead.)
    sensitivity = upper - lower
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = mechanisms.Laplace(epsilon=epsilon, sensitivity=sensitivity)
    private_sum = mech.randomise(true_sum)  # one number; use a tools function for arrays
    # Mechanisms do not charge an accountant themselves: spend the same epsilon explicitly.
    acc.spend(epsilon, 0)
    return private_sum, acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(100) % 30})
    private_sum, epsilon_spent = dp_sum_laplace(df, "value", bounds=(0, 50), epsilon=1.0)
    assert np.isfinite(private_sum)
    return {"epsilon": epsilon_spent}
