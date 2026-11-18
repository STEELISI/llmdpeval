import math

import numpy as np


def dp_two_by_two(df, exposure_column, exposure_value, outcome_column, outcome_value, epsilon, replace_one=True):
    """DP 2x2 table (exposure x outcome) from one Laplace histogram, plus odds ratio and relative risk.

    replace_one=True (default): bounded DP, n public; False: add/remove (n private).
    Returns (dict, epsilon_spent). The ratios are post-processing of the four noisy cells.
    """
    rng = np.random.default_rng()
    e = (df[exposure_column] == exposure_value).to_numpy()
    o = (df[outcome_column] == outcome_value).to_numpy()
    cells = np.array([np.sum(e & o), np.sum(e & ~o), np.sum(~e & o), np.sum(~e & ~o)], dtype=float)
    l1 = 2.0 if replace_one else 1.0         # one row is in exactly one cell
    scale = l1 / epsilon
    c11, c10, c01, c00 = np.maximum(cells + rng.laplace(0.0, scale, size=4), 0.0)
    epsilon_spent = l1 / scale
    noise_var = 2 * scale ** 2                # variance of one Laplace draw
    odds_ratio = ((c11 + 0.5) * (c00 + 0.5)) / ((c10 + 0.5) * (c01 + 0.5))   # continuity correction
    risk_exposed, risk_unexposed = (c11 + 0.5) / (c11 + c10 + 1.0), (c01 + 0.5) / (c01 + c00 + 1.0)
    # Noise-aware Wald test of log(OR) = 0: sampling variance sum 1/c plus DP noise variance sum v/c^2
    # (delta method), with v = 2*scale^2 the variance of the noise added to each cell.
    cs = np.array([c11, c10, c01, c00]) + 0.5
    se = math.sqrt(np.sum(1 / cs) + np.sum(noise_var / cs ** 2))
    p_value = math.erfc(abs(math.log(odds_ratio)) / se / math.sqrt(2))
    return {"cells": [c11, c10, c01, c00], "odds_ratio": odds_ratio, "log_or_se": se, "p_value": p_value,
            "relative_risk": risk_exposed / risk_unexposed}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"expo": ["y", "n"] * 200, "out": ["y", "y", "n", "n"] * 100})
    res, eps = dp_two_by_two(df, "expo", "y", "out", "y", epsilon=1.0)
    assert res["odds_ratio"] > 0
    return {"epsilon": float(eps)}
