import math

import numpy as np
from scipy.stats import norm


def cap_items(cell, items, max_items, sep, rng):
    """Distinct PUBLIC items of one unit, at most max_items chosen at random (looks only at this unit)."""
    own = list(dict.fromkeys(t for t in str(cell).split(sep) if t in items))
    return list(rng.permutation(own)[:max_items]) if len(own) > max_items else own


def dp_two_group_item_tables(df, group_column, groups, items_column, items, max_items, n_total, epsilon,
                             group_share=0.05, sep=";"):
    """For every PUBLIC item: in how many units of group a and of group b it is listed (one 2x2 table per item).
    One row = one unit; a unit lists many items in one delimited column. groups = (a, b) public labels,
    n_total public, max_items PUBLIC (a fixed rule, never from this data). Replace-one unit (bounded DP).
    Returns (dict of noisy counts and noise variances, epsilon_spent)."""
    rng = np.random.default_rng()
    items, col = list(items), {t: j for j, t in enumerate(items)}
    g_idx = df[group_column].map({groups[0]: 0, groups[1]: 1})   # rows in neither group are dropped
    keep = g_idx.notna().to_numpy()
    counts = np.zeros((2, len(items)))
    for g, cell in zip(g_idx[keep].astype(int), df[items_column].fillna("")[keep]):
        for t in cap_items(cell, col, max_items, sep, rng):
            counts[g, col[t]] += 1
    eps_g, eps_h = group_share * epsilon, (1 - group_share) * epsilon
    # Group sizes (n_a, n_b) as one vector: replace-one moves a unit from one group to the other, L1 <= 2.
    s_g = 2.0 / eps_g
    sizes = np.bincount(g_idx[keep].astype(int), minlength=2) + rng.laplace(0, s_g, 2)
    n_a, n_b = np.clip(sizes, 1, n_total - 1)
    # One (group, item) histogram: a capped unit adds at most max_items to it. Replace-one removes one unit
    # and adds another, maybe in the other group: L1 sensitivity 2 * max_items.
    s_h = 2.0 * max_items / eps_h
    noisy = counts + rng.laplace(0, s_h, counts.shape)
    eps_spent = 2.0 / s_g + 2.0 * max_items / s_h           # sequential composition of the two releases
    return {"A_a": noisy[0], "A_b": noisy[1], "n_a": float(n_a), "n_b": float(n_b),
            "var_cell": 2 * s_h ** 2, "var_n": 2 * s_g ** 2}, eps_spent


def noise_aware_tests(r, alpha=0.05, min_total_scales=5.0, n_mc=4000):
    """Post-processing only. Two-proportion score (Pearson) test per item that counts the DP noise;
    items tested = noisy total >= a public multiple of the noise scale; Bonferroni over the tested items."""
    A_a, A_b, n_a, n_b = np.maximum(r["A_a"], 0), np.maximum(r["A_b"], 0), r["n_a"], r["n_b"]
    n, pi, T = n_a + n_b, n_b / (n_a + n_b), A_a + A_b
    D = A_b - pi * T                                          # observed minus expected under "no difference"
    var_sampling = T * pi * (1 - pi) * (n - T) / (n - 1)
    sd = np.sqrt(np.maximum(var_sampling + ((1 - pi) ** 2 + pi ** 2) * (T / n) ** 2 * r["var_n"], 1e-12))
    # Laplace cell noise has heavy tails: p = mean over simulated cell noise of the normal tail (Bonferroni-safe).
    lap = np.sqrt(r["var_cell"] / 2) * np.random.default_rng().laplace(0.0, 1.0, (2, n_mc))
    e = (1 - pi) * lap[0] - pi * lap[1]                      # noise of D = A_b - pi * (A_a + A_b)
    d = np.abs(D)[:, None]
    p = (norm.sf((d - e) / sd[:, None]) + norm.cdf((-d - e) / sd[:, None])).mean(axis=1)
    tested = T >= min_total_scales * math.sqrt(r["var_cell"] / 2)
    significant = tested & (p <= alpha / max(tested.sum(), 1))
    ratio = ((A_b + 0.5) * (n_a - A_a + 0.5)) / ((n_b - A_b + 0.5) * (A_a + 0.5))
    return {"p": p, "tested": tested, "significant": significant, "ratio": ratio}


def _test():
    import pandas as pd
    items = [f"i{j}" for j in range(30)]
    n = 6000
    rng = np.random.default_rng()
    grp = rng.choice(["a", "b", "other"], n, p=[0.45, 0.45, 0.1])   # rows outside a/b must be dropped
    rows = []
    for g in grp:
        p = np.full(30, 0.05)
        p[:3] = 0.25 if g == "b" else 0.05                   # items i0-i2 differ between groups
        rows.append(";".join(t for t, hit in zip(items, rng.random(30) < p) if hit))
    df = pd.DataFrame({"grp": grp, "lst": rows})
    r, eps = dp_two_group_item_tables(df, "grp", ("a", "b"), "lst", items, max_items=4, n_total=n, epsilon=1.0)
    res = noise_aware_tests(r)
    assert len(res["p"]) == 30 and res["significant"][:3].all()
    return {"epsilon": float(eps)}
