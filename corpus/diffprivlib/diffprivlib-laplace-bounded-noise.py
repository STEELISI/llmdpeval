import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_mean_bounded_noise(df, column, bounds, epsilon, delta):
    """(epsilon, delta)-DP mean with bounded (truncated Laplacian) noise. bounds=(lower, upper) are public; n public.

    Returns (private_mean, (epsilon_spent, delta_spent)).
    """
    lower, upper = bounds
    values = np.clip(df[column].to_numpy(dtype=float), lower, upper)
    # Replace-one neighbours, n public: the mean moves by at most (upper - lower) / n.
    acc = BudgetAccountant(epsilon=epsilon, delta=delta)  # needs a nonzero delta budget
    acc.check(epsilon, delta)  # before the release: refuse an overspend
    mech = mechanisms.LaplaceBoundedNoise(epsilon=epsilon, delta=delta, sensitivity=(upper - lower) / len(values))
    private_mean = mech.randomise(float(values.mean()))
    acc.spend(epsilon, delta)
    return private_mean, tuple(acc.total())


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(500) % 60})
    m, (epsilon_spent, delta_spent) = dp_mean_bounded_noise(df, "value", (0, 60), epsilon=1.0, delta=1e-5)
    assert np.isfinite(m) and delta_spent == 1e-5
    return {"epsilon": epsilon_spent}
