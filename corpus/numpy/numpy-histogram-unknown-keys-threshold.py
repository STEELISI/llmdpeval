import numpy as np


def dp_counts_unknown_keys(df, column, epsilon, delta):
    """(epsilon, delta)-DP counts for categories NOT known in advance: Laplace + release threshold.

    Replace-one (bounded DP, n public), each record contributes to one category: a changed record
    lowers one key and raises another (L1 sensitivity 2), and up to two keys can appear or vanish.
    Returns ({key: noisy_count above threshold}, epsilon_spent); delta is the given delta.
    """
    rng = np.random.default_rng()
    keys, counts = np.unique(df[column].to_numpy(), return_counts=True)
    sens = 2.0                                    # replace-one: two keys move by 1 (1.0 under add/remove)
    scale = sens / epsilon
    noisy = counts + rng.laplace(0.0, scale, size=len(keys))
    # A key held by one record passes with prob 0.5 * exp(-(tau - 1) / scale); two keys can be at
    # risk under replace-one, so each gets delta / 2.
    tau = 1.0 + scale * np.log(1.0 / delta)
    released = {k: float(v) for k, v in zip(keys.tolist(), noisy) if v > tau}
    epsilon_spent = sens / scale
    assert epsilon_spent <= epsilon + 1e-9
    return released, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"tag": ["x"] * 80 + ["y"] * 40 + ["rare"]})
    res, eps = dp_counts_unknown_keys(df, "tag", epsilon=1.0, delta=1e-6)
    assert set(res) <= {"x", "y", "rare"}
    return {"epsilon": eps}
