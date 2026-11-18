import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import histogram


def dp_standardised_rates(df, group_column, groups, stratum_column, strata, population, std_weights,
                          group_a, group_b, epsilon, z=1.96):
    """DP event rates with PUBLIC denominators: stratum-specific, crude and directly standardised rates per group,
    and the standardised rate ratio group_a / group_b with a noise-aware confidence interval.
    Replace-one neighbours on the event rows (bounded DP).

    df: one row per event (private). groups, strata: PUBLIC lists. population[i][j]: PUBLIC population or
    person-time of group i, stratum j. std_weights: PUBLIC standard shares per stratum (sum 1).
    Returns (dict, epsilon_spent).
    """
    P, w = np.asarray(population, dtype=float), np.asarray(std_weights, dtype=float)
    gi = df[group_column].map({k: i for i, k in enumerate(groups)})
    sj = df[stratum_column].map({k: j for j, k in enumerate(strata)})
    ok = (gi.notna() & sj.notna()).to_numpy()
    cell = gi.to_numpy(dtype=float)[ok].astype(int) * len(strata) + sj.to_numpy(dtype=float)[ok].astype(int)
    acc = BudgetAccountant()
    # diffprivlib calibrates for add/remove; under replace-one a row moves between two cells: pass epsilon/2,
    # the guarantee is 2x the accountant total.
    counts, _ = histogram(cell, epsilon=epsilon / 2, bins=P.size, range=(-0.5, P.size - 0.5), accountant=acc)
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    d = counts.astype(float).reshape(P.shape)
    # Post-processing. Variance per cell = Poisson part + geometric noise variance 2a/(1-a)^2, a = exp(-epsilon/2)
    # (two-sided formula; the library truncates at 0, so this is slightly conservative near zero).
    a_ = np.exp(-epsilon / 2)
    var_d = d + 2 * a_ / (1 - a_) ** 2
    rate, crude = d / P, d.sum(axis=1) / P.sum(axis=1)
    asr = (w * rate).sum(axis=1)
    asr_var = (w ** 2 * var_d / P ** 2).sum(axis=1)
    a, b = groups.index(group_a), groups.index(group_b)
    rr = ci = None
    if asr[a] > 0 and asr[b] > 0:
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
    return {"epsilon": float(eps)}
