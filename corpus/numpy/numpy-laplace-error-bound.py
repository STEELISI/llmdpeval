import numpy as np


def dp_sum_with_interval(df, column, lower, upper, epsilon, beta=0.05, relation="replace_one"):
    """eps-DP clamped sum plus a (1 - beta) interval for the NOISE, from the Laplace tail.

    PUBLIC bounds [lower, upper]. The interval uses only public values (post-processing).
    Returns ({"sum", "low", "high", "beta"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    x = np.clip(df[column].to_numpy(dtype=float), lower, upper)
    # replace-one (default, n public): U - L; add/remove alternative: max(|L|, |U|)
    sens = upper - lower if relation == "replace_one" else max(abs(lower), abs(upper))
    scale = sens / epsilon
    noisy = float(x.sum() + rng.laplace(0.0, scale))
    half = scale * np.log(1 / beta)                 # P(|Lap(b)| >= b ln(1/beta)) = beta
    epsilon_spent = sens / scale
    assert epsilon_spent <= epsilon + 1e-9
    return {"sum": noisy, "low": noisy - half, "high": noisy + half, "beta": beta}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [3.0, 7.5, -2.0, 12.0, 5.0] * 20})
    res, eps = dp_sum_with_interval(df, "value", -5.0, 5.0, epsilon=1.0)
    assert res["low"] < res["sum"] < res["high"]
    return {"epsilon": eps}
