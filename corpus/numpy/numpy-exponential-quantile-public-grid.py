import numpy as np


def dp_quantile_on_grid(df, column, grid, qs, epsilon):
    """DP quantiles (0 < q < 1) chosen from a PUBLIC sorted grid of candidate values.

    Exponential mechanism with utility -|#{x <= g} - q*N|, sensitivity 1.
    Each quantile uses epsilon / len(qs). Returns ([values], epsilon_spent).
    """
    rng = np.random.default_rng()
    g = np.asarray(grid, dtype=float)
    x = np.sort(df[column].to_numpy(dtype=float))
    below = np.searchsorted(x, g, side="right")     # records <= each candidate
    sens_u = 1.0                                     # one record moves the utility by <= 1
    eps_each = epsilon / len(qs)
    out, spent = [], 0.0
    for q in qs:
        logits = eps_each * -np.abs(below - q * len(x)) / (2 * sens_u)
        p = np.exp(logits - logits.max())
        out.append(float(g[rng.choice(len(g), p=p / p.sum())]))
        spent += eps_each
    assert spent <= epsilon + 1e-9
    return out, spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": (np.arange(300) % 60).astype(float)})
    vals, eps = dp_quantile_on_grid(df, "value", np.arange(0, 101), [0.5], epsilon=1.0)
    assert 0 <= vals[0] <= 100
    return {"epsilon": eps}
