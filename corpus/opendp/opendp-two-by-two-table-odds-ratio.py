import math

import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


def dp_two_by_two(df, exposure_column, exposure_value, outcome_column, outcome_value, epsilon, replace_one=True):
    """DP 2x2 table (exposure x outcome) in one histogram, plus odds ratio and relative risk (post-processing).

    replace_one=True (default): bounded DP, n public, d_in = 2; False: add/remove-one, d_in = 1.
    The two levels of each variable are public (value / not value). Returns (dict, epsilon_spent).
    """
    cells = ["1|1", "1|0", "0|1", "0|0"]                 # exposure|outcome, public list
    e = (df[exposure_column] == exposure_value).map({True: "1", False: "0"})
    o = (df[outcome_column] == outcome_value).map({True: "1", False: "0"})
    labels = (e + "|" + o).tolist()
    d_in = 2 if replace_one else 1                       # replace-one moves a row between two cells
    space = (dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance())
    make = lambda s: space >> dp.t.then_count_by_categories(categories=cells) >> dp.m.then_laplace(scale=s)
    scale = dp.binary_search_param(make, d_in=d_in, d_out=epsilon)
    meas = make(scale)
    noise_var = 2 * scale ** 2                           # discrete Laplace: about 2 scale^2
    epsilon_spent = meas.map(d_in)
    c11, c10, c01, c00 = np.maximum(np.array(meas(labels)[:-1], dtype=float), 0.0)   # drop null bin, clamp
    odds_ratio = ((c11 + 0.5) * (c00 + 0.5)) / ((c10 + 0.5) * (c01 + 0.5))   # continuity correction
    risk_exposed, risk_unexposed = (c11 + 0.5) / (c11 + c10 + 1.0), (c01 + 0.5) / (c01 + c00 + 1.0)
    # Noise-aware Wald test of log(OR) = 0: sampling variance sum 1/c plus DP noise variance sum v/c^2.
    cs = np.array([c11, c10, c01, c00]) + 0.5
    se = math.sqrt(np.sum(1 / cs) + np.sum(noise_var / cs ** 2))
    p_value = math.erfc(abs(math.log(odds_ratio)) / se / math.sqrt(2))
    return {"cells": dict(zip(cells, [c11, c10, c01, c00])), "odds_ratio": odds_ratio, "log_or_se": se,
            "p_value": p_value, "relative_risk": risk_exposed / risk_unexposed}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"expo": ["y", "n"] * 200, "out": ["y", "y", "n", "n"] * 100})
    res, eps = dp_two_by_two(df, "expo", "y", "out", "y", epsilon=1.0)
    assert res["odds_ratio"] > 0
    return {"epsilon": float(eps)}
