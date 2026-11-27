import numpy as np


def dp_quantiles(df, column, qs, lower, upper, epsilon):
    """DP quantiles (each 0 < q < 1) of a column with PUBLIC bounds, exponential mechanism over intervals.

    Each quantile runs the mechanism once with epsilon / len(qs). Returns ([private_q...], epsilon_spent).
    """
    rng = np.random.default_rng()
    x = np.clip(df[column].to_numpy(dtype=float), lower, upper)
    n = len(x)
    z = np.concatenate([[lower], np.sort(x), [upper]])       # n + 1 intervals [z_j, z_j+1]
    j = np.arange(n + 1)                                     # interval j has j points to its left
    with np.errstate(divide="ignore"):
        log_len = np.log(np.diff(z))                         # zero-length intervals get probability 0
    sens_u = 1.0                                             # utility sensitivity (replace-one; <= 1 add/remove)
    eps_each = epsilon / len(qs)
    results, epsilon_spent = [], 0.0
    for q in qs:
        u = -np.abs(j - q * n)
        logits = log_len + eps_each * u / (2 * sens_u)
        p = np.exp(logits - np.max(logits))
        p /= p.sum()
        k = rng.choice(n + 1, p=p)
        results.append(float(rng.uniform(z[k], z[k + 1])))  # uniform inside the chosen interval
        epsilon_spent += eps_each                            # one exponential mechanism per quantile
    assert epsilon_spent <= epsilon + 1e-9
    return results, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(200, dtype=float) % 37})
    qs, epsilon_spent = dp_quantiles(df, "value", [0.25, 0.5, 0.75], 0.0, 50.0, epsilon=1.0)
    assert all(0.0 <= v <= 50.0 for v in qs)
    return {"epsilon": epsilon_spent}
