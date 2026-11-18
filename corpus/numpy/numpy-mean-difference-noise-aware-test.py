import math

import numpy as np


def dp_mean_difference_test(df, group_column, group_a, group_b, value_column, bounds, epsilon):
    """Difference of two group means with a noise-aware z-test. Group sizes are PRIVATE: each group's
    size, sum and sum of squares are released together as one noisy vector. Replace-one neighbours
    (bounded DP, n public). bounds: PUBLIC (low, high) for the value column.

    Returns ({"mean_a", "mean_b", "diff", "se", "p", "noisy_sizes"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    lo, hi = bounds
    v = (df[value_column].astype(float).clip(lo, hi) - lo) / (hi - lo)     # in [0, 1] by construction
    stats = []
    for g in (group_a, group_b):
        x = v[df[group_column] == g].to_numpy()
        stats += [len(x), x.sum(), (x * x).sum()]                         # (n_g, S_g, Q_g)
    # One row adds (1, x, x^2) to ONE group: L1 <= 3. Replace-one can move it from one group to the
    # other (remove + add), so the L1 sensitivity of the 6-vector is 6.
    sens = 6.0
    scale = sens / epsilon
    n_a, s_a, q_a, n_b, s_b, q_b = np.array(stats, float) + rng.laplace(0.0, scale, 6)
    epsilon_spent = sens / scale
    noise_var = 2 * scale ** 2                                            # variance of one Laplace draw
    out, var = {}, 0.0                                                    # post-processing from here on
    for name, n, s, q in (("a", n_a, s_a, q_a), ("b", n_b, s_b, q_b)):
        n = max(n, 2.0)                                                   # public floor for tiny groups
        m = min(max(s / n, 0.0), 1.0)                                     # clip to the valid range
        sample_var = min(max(q / n - m * m, 0.0), 0.25) / n               # sampling variance of the mean
        dp_var = noise_var * (1 + m * m) / n ** 2                         # delta method: noise in S and n
        var += sample_var + dp_var
        out[f"mean_{name}"] = lo + m * (hi - lo)
        out.setdefault("noisy_sizes", {})[name] = n
    out["diff"] = out["mean_a"] - out["mean_b"]
    out["se"] = math.sqrt(var) * (hi - lo)
    out["p"] = math.erfc(abs(out["diff"] / out["se"]) / math.sqrt(2))
    return out, epsilon_spent


def _test():
    rng = np.random.default_rng()
    import pandas as pd
    m = 3000
    g = rng.choice(["p", "q", "r"], m, p=[0.1, 0.6, 0.3])
    val = np.clip(rng.normal(np.where(g == "p", 3.5, 3.0), 1.0), 1, 5)
    df = pd.DataFrame({"grp": g, "score": val})
    res, eps = dp_mean_difference_test(df, "grp", "p", "q", "score", (1.0, 5.0), epsilon=1.0)
    assert res["se"] > 0 and 0 <= res["p"] <= 1
    return {"epsilon": float(eps)}
