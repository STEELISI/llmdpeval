import numpy as np


def dp_variance(df, column, lower, upper, epsilon, relation="replace_one", public_n=None):
    """DP variance of a column clamped to PUBLIC bounds, from noisy first and second moments.

    Returns (private_variance, epsilon_spent). Budget split evenly over the released statistics.
    """
    rng = np.random.default_rng()
    c, r = (lower + upper) / 2, (upper - lower) / 2          # public midpoint and half-width
    z = np.clip(df[column].to_numpy(dtype=float), lower, upper) - c   # every z in [-r, r]
    s1, s2 = np.sum(z), np.sum(z ** 2)
    if relation == "add_remove":                             # sens: s1 -> r, s2 -> r^2, count -> 1
        sens, stats = [r, r ** 2, 1.0], [s1, s2, len(z)]
    else:                                                    # replace-one, N public: s1 -> 2r, s2 -> r^2
        assert public_n is not None and public_n == len(z), "replace-one needs a public N"
        sens, stats = [2 * r, r ** 2], [s1, s2]
    eps_each = epsilon / len(sens)
    scales = [d / eps_each for d in sens]
    noisy = [s + rng.laplace(0.0, b) for s, b in zip(stats, scales)]
    n_hat = max(noisy[2], 1.0) if relation == "add_remove" else public_n
    m = noisy[0] / n_hat                                     # post-processing from here on
    var = noisy[1] / n_hat - m ** 2
    epsilon_spent = sum(d / b for d, b in zip(sens, scales))  # sequential composition
    assert epsilon_spent <= epsilon + 1e-9
    return float(np.clip(var, 0.0, r ** 2)), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [3.0, 7.5, -2.0, 12.0, 5.0] * 40})
    v1, e1 = dp_variance(df, "value", -5.0, 5.0, epsilon=1.0, public_n=len(df))   # bounds straddle 0
    v2, e2 = dp_variance(df, "value", -5.0, 5.0, epsilon=1.0, relation="add_remove")
    assert 0.0 <= v1 <= 25.0 and 0.0 <= v2 <= 25.0
    return {"epsilon": float(max(e1, e2))}
