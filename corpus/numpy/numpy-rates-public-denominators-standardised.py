import numpy as np


def dp_standardised_rates(df, group_column, groups, stratum_column, strata, population, std_weights,
                          group_a, group_b, epsilon, relation="replace_one", z=1.96):
    """DP event rates with PUBLIC denominators: stratum-specific, crude and directly standardised rates per group,
    and the standardised rate ratio group_a / group_b with a noise-aware confidence interval.

    df: one row per event (private). groups, strata: PUBLIC lists. population[i][j]: PUBLIC population or
    person-time of group i, stratum j. std_weights: PUBLIC standard shares per stratum (sum 1).
    Returns (dict, epsilon_spent).
    """
    rng = np.random.default_rng()
    P, w = np.asarray(population, dtype=float), np.asarray(std_weights, dtype=float)
    gi = df[group_column].map({k: i for i, k in enumerate(groups)})
    sj = df[stratum_column].map({k: j for j, k in enumerate(strata)})
    ok = (gi.notna() & sj.notna()).to_numpy()
    cell = gi.to_numpy(dtype=float)[ok].astype(int) * len(strata) + sj.to_numpy(dtype=float)[ok].astype(int)
    counts = np.bincount(cell, minlength=P.size).reshape(P.shape).astype(float)
    # One event row = one cell: L1 sensitivity 1 (add/remove), 2 (replace-one: the row moves to another cell).
    l1 = 1.0 if relation == "add_remove" else 2.0
    scale = l1 / epsilon
    d = counts + rng.laplace(0.0, scale, size=counts.shape)
    epsilon_spent = l1 / scale
    assert epsilon_spent <= epsilon + 1e-9
    # Post-processing. Variance per cell = Poisson part (estimated from the noisy count) + Laplace variance.
    var_d = np.maximum(d, 0.0) + 2 * scale ** 2
    rate, crude = d / P, d.sum(axis=1) / P.sum(axis=1)
    asr = (w * rate).sum(axis=1)
    asr_var = (w ** 2 * var_d / P ** 2).sum(axis=1)
    a, b = groups.index(group_a), groups.index(group_b)
    rr = ci = None
    if asr[a] > 0 and asr[b] > 0:                       # log ratio undefined otherwise
        rr = asr[a] / asr[b]
        se = np.sqrt(asr_var[a] / asr[a] ** 2 + asr_var[b] / asr[b] ** 2)
        ci = (rr * np.exp(-z * se), rr * np.exp(z * se))
    return {"stratum_rates": rate, "crude": crude, "standardised": asr, "standardised_se": np.sqrt(asr_var),
            "rate_ratio": rr, "rate_ratio_ci": ci}, epsilon_spent


def _test():
    import pandas as pd
    rng = np.random.default_rng()
    df = pd.DataFrame({"grp": rng.choice(["a", "b"], 3000), "band": rng.choice(["y", "m", "o"], 3000)})
    pop = [[50000, 40000, 30000], [60000, 30000, 20000]]
    res, eps = dp_standardised_rates(df, "grp", ["a", "b"], "band", ["y", "m", "o"], pop, [0.4, 0.35, 0.25],
                                     "a", "b", epsilon=1.0)
    assert res["rate_ratio"] is not None
    return {"epsilon": eps}
