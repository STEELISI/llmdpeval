import numpy as np


def dp_count_where(df, column, value, epsilon):
    """DP count of rows where df[column] == value, Laplace mechanism.

    Returns (private_count, epsilon_spent). Δ = 1 under add/remove and under replace-one.
    """
    rng = np.random.default_rng()                    # OS entropy; never a fixed seed in a release
    true_count = int(np.sum(df[column].to_numpy() == value))
    sensitivity = 1.0                                # one record changes a predicate count by at most 1
    scale = sensitivity / epsilon                    # Laplace parameter b, not the standard deviation
    private_count = true_count + rng.laplace(loc=0.0, scale=scale)
    private_count = max(0, round(private_count))     # post-processing on the noisy value only
    epsilon_spent = sensitivity / scale
    return private_count, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    private_count, epsilon_spent = dp_count_where(df, "group", "a", epsilon=1.0)
    assert isinstance(private_count, int)
    return {"epsilon": float(epsilon_spent)}
