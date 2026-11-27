import math

import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import histogram


def dp_two_by_two(df, exposure_column, exposure_value, outcome_column, outcome_value, epsilon):
    """DP 2x2 table (exposure x outcome) from one diffprivlib histogram over 4 cell codes, replace-one (bounded DP, n public).

    Returns (dict with cells, odds_ratio, relative_risk; epsilon_spent). Ratios are post-processing.
    """
    e = (df[exposure_column] == exposure_value).to_numpy()
    o = (df[outcome_column] == outcome_value).to_numpy()
    codes = 2 * (~e).astype(int) + (~o).astype(int)   # 0=(1,1) 1=(1,0) 2=(0,1) 3=(0,0)
    acc = BudgetAccountant()
    # diffprivlib calibrates histograms for add/remove (sensitivity 1 per bin). Under replace-one a changed row
    # leaves one bin and enters another, so pass epsilon/2; the guarantee is 2x what the accountant recorded.
    cells, _ = histogram(codes, epsilon=epsilon / 2, bins=4, range=(-0.5, 3.5), accountant=acc)   # public range
    c11, c10, c01, c00 = np.maximum(cells.astype(float), 0.0)
    noise_var = 2 * (1 / (epsilon / 2)) ** 2   # geometric noise, sensitivity 1 at epsilon/2: ~2 scale^2
    odds_ratio = ((c11 + 0.5) * (c00 + 0.5)) / ((c10 + 0.5) * (c01 + 0.5))   # continuity correction
    risk_exposed, risk_unexposed = (c11 + 0.5) / (c11 + c10 + 1.0), (c01 + 0.5) / (c01 + c00 + 1.0)
    # Noise-aware Wald test of log(OR) = 0: sampling variance sum 1/c plus DP noise variance sum v/c^2
    # (delta method), with v = ~2*scale^2 the variance of the noise added to each cell.
    cs = np.array([c11, c10, c01, c00]) + 0.5
    se = math.sqrt(np.sum(1 / cs) + np.sum(noise_var / cs ** 2))
    p_value = math.erfc(abs(math.log(odds_ratio)) / se / math.sqrt(2))
    return {"cells": [c11, c10, c01, c00], "odds_ratio": odds_ratio, "log_or_se": se, "p_value": p_value,
            "relative_risk": risk_exposed / risk_unexposed}, 2 * acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"expo": ["y", "n"] * 200, "out": ["y", "y", "n", "n"] * 100})
    res, eps = dp_two_by_two(df, "expo", "y", "out", "y", epsilon=1.0)
    assert res["odds_ratio"] > 0
    return {"epsilon": float(eps)}
