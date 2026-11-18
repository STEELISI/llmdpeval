import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_mean_staircase(df, column, bounds, epsilon):
    """DP mean with the Staircase mechanism. bounds=(lower, upper) are public; n is public (bounded DP).

    Returns (private_mean, epsilon_spent). Output is clamped to the bounds as post-processing.
    """
    lower, upper = bounds
    values = np.clip(df[column].to_numpy(dtype=float), lower, upper)
    # Replace-one neighbours, n public: the mean moves by at most (upper - lower) / n.
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = mechanisms.Staircase(epsilon=epsilon, sensitivity=(upper - lower) / len(values))
    private_mean = float(np.clip(mech.randomise(float(values.mean())), lower, upper))
    acc.spend(epsilon, 0)  # mechanisms never charge an accountant themselves
    return private_mean, acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(300) % 40})
    m, epsilon_spent = dp_mean_staircase(df, "value", (0, 40), epsilon=1.0)
    assert 0 <= m <= 40
    return {"epsilon": epsilon_spent}
