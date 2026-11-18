import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_integer_total_folded(df, column, value_bounds, output_bounds, epsilon):
    """DP integer total of df[column], folded into the public output range output_bounds=(L, U).

    value_bounds=(lower, upper) are public integers used to clip each value. Returns (private_total, epsilon_spent).
    """
    lower, upper = value_bounds
    true_total = int(np.clip(df[column].to_numpy(), lower, upper).sum())
    # Replace-one neighbours: one clipped integer value moves by at most upper - lower.
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = mechanisms.GeometricFolded(epsilon=epsilon, sensitivity=upper - lower,
                                      lower=output_bounds[0], upper=output_bounds[1])
    private_total = mech.randomise(true_total)  # noise outside the range is reflected back
    acc.spend(epsilon, 0)  # mechanisms never charge an accountant themselves
    return int(private_total), acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"units": np.arange(100) % 4})
    total, epsilon_spent = dp_integer_total_folded(df, "units", (0, 4), (0, 400), epsilon=1.0)
    assert 0 <= total <= 400
    return {"epsilon": epsilon_spent}
