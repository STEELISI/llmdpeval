import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms.transforms import RoundedInteger


def dp_bounded_integer_total(df, column, value_bounds, output_bounds, epsilon):
    """DP integer total that stays inside public output_bounds, via LaplaceBoundedDomain + RoundedInteger.

    value_bounds=(lower, upper) clip each value (public). Returns (private_int_total, epsilon_spent).
    """
    lower, upper = value_bounds
    true_total = float(np.clip(df[column].to_numpy(dtype=float), lower, upper).sum())
    # Replace-one neighbours: one clipped value moves by at most upper - lower.
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    inner = mechanisms.LaplaceBoundedDomain(epsilon=epsilon, sensitivity=upper - lower,
                                            lower=output_bounds[0], upper=output_bounds[1])
    private_total = RoundedInteger(inner).randomise(true_total)  # rounding is post-processing
    acc.spend(epsilon, 0)  # mechanisms never charge an accountant themselves
    return private_total, acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"units": np.arange(100) % 5})
    total, epsilon_spent = dp_bounded_integer_total(df, "units", (0, 5), (0, 500), epsilon=1.0)
    assert isinstance(total, int) and 0 <= total <= 500
    return {"epsilon": epsilon_spent}
