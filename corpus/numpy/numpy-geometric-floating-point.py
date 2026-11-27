import numpy as np


def dp_count_geometric(df, column, value, epsilon):
    """DP integer count of rows where df[column] == value, two-sided geometric mechanism.

    Integer noise avoids the floating-point attack on Laplace doubles (Mironov 2012).
    Returns (private_count, epsilon_spent).
    """
    rng = np.random.default_rng()
    true_count = int(np.sum(df[column].to_numpy() == value))
    sensitivity = 1                                  # single count: Δ = 1 under replace-one (and add/remove)
    p = -np.expm1(-epsilon / sensitivity)            # p = 1 - e^(-ε/Δ), accurate for small ε/Δ
    # Difference of two independent geometric draws: P(k) ∝ e^(-ε|k|/Δ) on the integers.
    noise = int(rng.geometric(p) - rng.geometric(p))
    private_count = max(0, true_count + noise)       # post-processing: clip at 0
    epsilon_spent = float(-sensitivity * np.log1p(-p))   # recover ε from the p actually used
    return private_count, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    private_count, epsilon_spent = dp_count_geometric(df, "group", "a", epsilon=1.0)
    assert isinstance(private_count, int)
    return {"epsilon": epsilon_spent}
