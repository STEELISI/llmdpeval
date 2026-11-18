import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import quantile


def dp_group_quantile_gaps(df, group_column, groups, value_column, bounds, alphas, group_a, group_b, epsilon):
    """DP quantiles (e.g. alphas 0.1, 0.5, 0.9) per group of a PUBLIC group list, the gap group_a - group_b at
    each alpha, and each group's ratio of its highest to its lowest quantile (e.g. p90/p10).
    Replace-one neighbours (bounded DP, n public): a changed row can leave one group and join another.

    bounds: PUBLIC (L, U) for the value column. Returns ({"quantiles", "gaps", "ratios"}, epsilon_spent).
    """
    accs, quantiles = {}, {}
    for g in groups:
        x = df.loc[df[group_column] == g, value_column].to_numpy(dtype=float)
        accs[g] = BudgetAccountant()
        # Groups are disjoint, but under replace-one one row can change TWO groups' data (it leaves one and
        # joins another), so each group gets epsilon/2. The list of alphas is split evenly inside the call.
        q = quantile(x, list(alphas), epsilon=epsilon / 2, bounds=bounds, accountant=accs[g])
        quantiles[g] = dict(zip(sorted(alphas), np.sort(q).tolist()))   # sorting is post-processing
    # Worst case = the two groups a changed row can touch: the sum of the two largest per-group totals.
    totals = sorted((a.total()[0] for a in accs.values()), reverse=True)
    epsilon_spent = sum(totals[:2])
    assert epsilon_spent <= epsilon + 1e-9
    lo, hi = min(alphas), max(alphas)
    gaps = {a: quantiles[group_a][a] - quantiles[group_b][a] for a in alphas}
    ratios = {g: (q[hi] / q[lo] if q[lo] > 0 else None) for g, q in quantiles.items()}
    return {"quantiles": quantiles, "gaps": gaps, "ratios": ratios}, epsilon_spent


def _test():
    import pandas as pd
    rng = np.random.default_rng()
    df = pd.DataFrame({"grp": rng.choice(["a", "b"], 2000), "value": rng.lognormal(3.0, 0.5, 2000)})
    res, eps = dp_group_quantile_gaps(df, "grp", ["a", "b", "c"], "value", (1.0, 200.0), [0.1, 0.5, 0.9],
                                      "a", "b", epsilon=1.0)
    assert set(res["gaps"]) == {0.1, 0.5, 0.9}
    return {"epsilon": float(eps)}
