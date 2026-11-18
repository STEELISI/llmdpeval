import numpy as np


def dp_sum(df, column, lower, upper, epsilon, relation="replace_one"):
    """DP sum of a numeric column clamped to PUBLIC bounds [lower, upper] (never from the data).

    Returns (private_sum, epsilon_spent).
    """
    rng = np.random.default_rng()
    x = np.clip(df[column].to_numpy(dtype=float), lower, upper)   # clamp, don't drop rows
    true_sum = float(np.sum(x))
    # replace-one (default, N public): one value moves from L to U at worst;
    # add/remove (alternative): one record contributes at most max(|L|, |U|).
    if relation == "replace_one":
        sensitivity = upper - lower
    else:
        sensitivity = max(abs(lower), abs(upper))
    scale = sensitivity / epsilon
    private_sum = true_sum + rng.laplace(loc=0.0, scale=scale)
    epsilon_spent = sensitivity / scale
    return private_sum, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [3.0, 7.5, -2.0, 12.0, 5.0] * 20})
    private_sum, epsilon_spent = dp_sum(df, "value", -5.0, 5.0, epsilon=1.0)   # bounds straddle 0
    assert np.isfinite(private_sum)
    return {"epsilon": float(epsilon_spent)}
