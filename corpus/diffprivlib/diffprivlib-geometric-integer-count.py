from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_count_geometric(df, column, value, epsilon, sensitivity=1, upper=None):
    """DP count of rows with df[column] == value, geometric (discrete Laplace) noise, integer output.

    sensitivity: 1 if each person has one row, else the public per-person contribution cap (an int).
    upper: optional public population size; then the output is truncated to [0, upper].
    Returns (private_count, epsilon_spent).
    """
    true_count = int((df[column] == value).sum())
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    if upper is None:
        mech = mechanisms.Geometric(epsilon=epsilon, sensitivity=sensitivity)
    else:
        mech = mechanisms.GeometricTruncated(epsilon=epsilon, sensitivity=sensitivity, lower=0, upper=int(upper))
    private_count = mech.randomise(true_count)
    acc.spend(epsilon, 0)  # mechanisms do not charge an accountant themselves
    return int(private_count), acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    private_count, epsilon_spent = dp_count_geometric(df, "group", "a", epsilon=1.0, upper=1000)
    assert 0 <= private_count <= 1000
    return {"epsilon": epsilon_spent}
