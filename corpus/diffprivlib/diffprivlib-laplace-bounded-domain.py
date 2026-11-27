import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_mean_bounded_domain(df, column, bounds, epsilon):
    """DP mean whose output must stay inside public bounds=(lower, upper), via LaplaceBoundedDomain.

    Bounded DP (replace-one): the row count n is public. Returns (private_mean, epsilon_spent).
    """
    lower, upper = bounds
    values = np.clip(df[column].to_numpy(dtype=float), lower, upper)
    true_mean = float(values.mean())  # inside [lower, upper] by construction, as randomise requires
    sensitivity = (upper - lower) / len(values)  # one replaced row moves the mean by at most (upper - lower) / n
    # LaplaceTruncated / LaplaceFolded take the same arguments (clamp / reflect instead of resampling).
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = mechanisms.LaplaceBoundedDomain(epsilon=epsilon, sensitivity=sensitivity, lower=lower, upper=upper)
    private_mean = mech.randomise(true_mean)
    acc.spend(epsilon, 0)  # mechanisms do not charge an accountant themselves
    return private_mean, acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(100) % 30})
    private_mean, epsilon_spent = dp_mean_bounded_domain(df, "value", bounds=(0, 50), epsilon=1.0)
    assert 0 <= private_mean <= 50
    return {"epsilon": epsilon_spent}
