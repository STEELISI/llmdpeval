import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_sum_snapping(df, column, value_bounds, output_bounds, epsilon):
    """DP sum with the Snapping mechanism (floating-point-safe Laplace). All bounds are public.

    value_bounds clip each value; output_bounds=(L, U) clamp input and output. Returns (private_sum, epsilon_spent).
    """
    lower, upper = value_bounds
    true_sum = float(np.clip(df[column].to_numpy(dtype=float), lower, upper).sum())
    # Replace-one neighbours: one clipped value moves the sum by at most upper - lower.
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = mechanisms.Snapping(epsilon=epsilon, sensitivity=upper - lower,
                               lower=output_bounds[0], upper=output_bounds[1])
    private_sum = float(mech.randomise(true_sum))  # coarse (snapped) output is expected
    acc.spend(epsilon, 0)  # mechanisms never charge an accountant themselves
    return private_sum, acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(50) % 10 / 10})
    s, epsilon_spent = dp_sum_snapping(df, "value", (0, 1), (0, 50), epsilon=1.0)
    assert 0 <= s <= 50
    return {"epsilon": epsilon_spent}
