import math

import numpy as np


def _analytic_delta(sigma, sens2, epsilon):
    """Balle & Wang 2018 Thm. 8: the delta achieved by N(0, sigma^2) noise at this epsilon."""
    phi = lambda t: 0.5 * math.erfc(-t / math.sqrt(2))
    a, b = sens2 / (2 * sigma), epsilon * sigma / sens2
    return phi(a - b) - math.exp(epsilon) * phi(-a - b)


def dp_gaussian_sums(df, columns, bounds, epsilon, delta, relation="replace_one"):
    """(epsilon, delta)-DP vector of column sums, Gaussian noise calibrated to the L2 sensitivity.

    bounds: {column: (lower, upper)}, public. Choose delta << 1/N. Returns ({column: sum}, epsilon_spent).
    """
    rng = np.random.default_rng()
    lo = np.array([bounds[c][0] for c in columns], dtype=float)
    hi = np.array([bounds[c][1] for c in columns], dtype=float)
    x = np.clip(df[columns].to_numpy(dtype=float), lo, hi)
    # L2 sensitivity of the sum vector: replace-one (default, n public) moves each column by at most
    # U - L; add/remove (alternative) adds at most max(|L|, |U|) per column.
    per_col = hi - lo if relation == "replace_one" else np.maximum(np.abs(lo), np.abs(hi))
    sens2 = float(np.sqrt(np.sum(per_col ** 2)))
    s_lo, s_hi = 1e-6 * sens2, sens2                     # bisection for the smallest valid sigma
    while _analytic_delta(s_hi, sens2, epsilon) > delta:
        s_hi *= 2
    for _ in range(100):
        mid = (s_lo + s_hi) / 2
        s_lo, s_hi = (s_lo, mid) if _analytic_delta(mid, sens2, epsilon) <= delta else (mid, s_hi)
    sigma = s_hi                                         # standard deviation, not variance
    assert _analytic_delta(sigma, sens2, epsilon) <= delta
    noisy = x.sum(axis=0) + rng.normal(loc=0.0, scale=sigma, size=len(columns))
    return dict(zip(columns, noisy.tolist())), epsilon   # (epsilon, delta) verified by the assert


def _test():
    import pandas as pd
    df = pd.DataFrame({"a": [1.0, 4.0, 2.5, 9.0] * 25, "b": [-3.0, 0.5, 2.0, 1.0] * 25})
    res, epsilon_spent = dp_gaussian_sums(df, ["a", "b"], {"a": (0.0, 10.0), "b": (-5.0, 5.0)},
                                          epsilon=1.0, delta=1e-5)
    assert set(res) == {"a", "b"}
    return {"epsilon": epsilon_spent}
