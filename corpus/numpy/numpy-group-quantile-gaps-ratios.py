import numpy as np


def dp_group_quantile_gaps(df, group_column, groups, value_column, grid, alphas, group_a, group_b, epsilon,
                           relation="replace_one"):
    """DP quantiles (e.g. alphas 0.1, 0.5, 0.9) per group of a PUBLIC group list by the exponential mechanism over
    a PUBLIC candidate grid, the gap group_a - group_b at each alpha, and each group's ratio of its highest to its
    lowest quantile (e.g. p90/p10). Returns ({"quantiles", "gaps", "ratios"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    G = np.asarray(grid, dtype=float)
    # Replace-one: a changed row can leave one group and join another, so two groups' releases change; each group
    # gets epsilon/2 (add/remove: one group changes, it gets epsilon). Within a group, alphas split evenly.
    per_group = epsilon / 2 if relation == "replace_one" else epsilon
    eps_q = per_group / len(alphas)
    quantiles = {}
    for g in groups:
        x = np.sort(df.loc[df[group_column] == g, value_column].to_numpy(dtype=float))
        below = np.searchsorted(x, G, side="right")          # records <= each candidate
        qs = []
        for a in alphas:
            u = -np.abs(below - a * len(x))                  # utility sensitivity 1 for one changed row
            logits = eps_q * u / 2.0
            p = np.exp(logits - logits.max())
            qs.append(G[rng.choice(len(G), p=p / p.sum())])
        quantiles[g] = dict(zip(sorted(alphas), sorted(qs)))  # sorting is post-processing
    epsilon_spent = (2 if relation == "replace_one" else 1) * eps_q * len(alphas)   # sequential composition
    assert epsilon_spent <= epsilon + 1e-9
    lo, hi = min(alphas), max(alphas)
    gaps = {a: quantiles[group_a][a] - quantiles[group_b][a] for a in alphas}
    ratios = {g: (q[hi] / q[lo] if q[lo] > 0 else None) for g, q in quantiles.items()}
    return {"quantiles": quantiles, "gaps": gaps, "ratios": ratios}, epsilon_spent


def _test():
    import pandas as pd
    rng = np.random.default_rng()
    df = pd.DataFrame({"grp": rng.choice(["a", "b"], 2000), "value": rng.lognormal(3.0, 0.5, 2000)})
    grid = np.geomspace(1.0, 200.0, 120)
    res, eps = dp_group_quantile_gaps(df, "grp", ["a", "b", "c"], "value", grid, [0.1, 0.5, 0.9], "a", "b", 1.0)
    assert set(res["gaps"]) == {0.1, 0.5, 0.9}
    return {"epsilon": float(eps)}
