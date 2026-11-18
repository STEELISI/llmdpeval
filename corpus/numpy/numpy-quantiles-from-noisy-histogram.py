import numpy as np


def dp_quantiles_histogram(df, column, edges, qs, epsilon, relation="replace_one"):
    """DP quantiles read from ONE Laplace histogram over PUBLIC bin edges.

    All quantiles come from the same noisy histogram (post-processing), so the cost is epsilon once.
    relation: "replace_one" (bounded DP, n public; default) or "add_remove".
    Returns ([quantile values], epsilon_spent).
    """
    rng = np.random.default_rng()
    e = np.asarray(edges, dtype=float)
    x = np.clip(df[column].to_numpy(dtype=float), e[0], e[-1])   # every record lands in a bin
    counts = np.histogram(x, bins=e)[0].astype(float)
    l1 = 1.0 if relation == "add_remove" else 2.0                 # L1 sensitivity of the bin vector
    scale = l1 / epsilon
    noisy = np.clip(counts + rng.laplace(0.0, scale, size=len(counts)), 0.0, None)
    cdf = np.cumsum(noisy) / max(noisy.sum(), 1e-12)               # post-processing from here on
    out = []
    for q in qs:
        b = min(int(np.searchsorted(cdf, q)), len(counts) - 1)
        lo_cdf = cdf[b - 1] if b > 0 else 0.0
        frac = (q - lo_cdf) / max(cdf[b] - lo_cdf, 1e-12)          # linear interpolation in bin b
        out.append(float(e[b] + np.clip(frac, 0, 1) * (e[b + 1] - e[b])))
    epsilon_spent = l1 / scale
    assert epsilon_spent <= epsilon + 1e-9
    return out, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": (np.arange(400) % 80).astype(float)})
    vals, eps = dp_quantiles_histogram(df, "value", np.linspace(0, 100, 21), [0.25, 0.5, 0.75], 1.0)
    assert all(0 <= v <= 100 for v in vals)
    return {"epsilon": eps}
