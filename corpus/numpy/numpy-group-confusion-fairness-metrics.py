import numpy as np


def dp_group_confusion(df, true_column, pred_column, group_column, positive_value, group_values, epsilon):
    """One Laplace release of the (group x true label x predicted label) count table of a binary classifier on
    PRIVATE held-out rows. group_values: PUBLIC list (rows outside it are dropped). Replace-one, n public;
    group sizes are private. Returns (noisy table of shape (G, 2, 2) as [group, true, pred], epsilon_spent)."""
    rng = np.random.default_rng()
    g = df[group_column].map({v: i for i, v in enumerate(group_values)}).to_numpy(dtype=float, na_value=-1)
    t = (df[true_column] == positive_value).to_numpy(dtype=int)
    p = (df[pred_column] == positive_value).to_numpy(dtype=int)
    ok = g >= 0
    table = np.zeros((len(group_values), 2, 2))
    np.add.at(table, (g[ok].astype(int), t[ok], p[ok]), 1)
    sensitivity = 2.0          # replace-one: one row leaves one cell and enters another (L1 = 2)
    scale = sensitivity / epsilon
    noisy = table + rng.laplace(0.0, scale, table.shape)
    epsilon_spent = sensitivity / scale   # NumPy has no accountant: Laplace(L1 / eps) is eps-DP
    return noisy, epsilon_spent


def fairness_metrics(noisy, group_values, unprivileged, privileged, n_total):
    """Post-processing only: overall accuracy (n public), precision, recall, and the per-group gaps
    (unprivileged minus privileged) with NOISY denominators. Rates clipped to [0, 1]."""
    m = np.maximum(noisy, 0.0)
    rate = lambda num, den: float(np.clip(num / max(den, 1.0), 0, 1))
    tot = m.sum(axis=0)                                   # [true, pred] over all groups
    out = {"accuracy": rate(tot[0, 0] + tot[1, 1], n_total),
           "precision": rate(tot[1, 1], tot[:, 1].sum()), "recall": rate(tot[1, 1], tot[1].sum())}
    per = {}
    for v in (unprivileged, privileged):
        c = m[group_values.index(v)]
        per[v] = {"ppr": rate(c[:, 1].sum(), c.sum()), "tpr": rate(c[1, 1], c[1].sum()),
                  "fpr": rate(c[0, 1], c[0].sum())}
    u, r = per[unprivileged], per[privileged]
    out["statistical_parity_difference"] = u["ppr"] - r["ppr"]
    out["equal_opportunity_difference"] = u["tpr"] - r["tpr"]
    out["average_odds_difference"] = 0.5 * ((u["fpr"] - r["fpr"]) + (u["tpr"] - r["tpr"]))
    out["disparate_impact"] = u["ppr"] / max(r["ppr"], 1e-9)
    return out


def _test():
    import pandas as pd
    rng = np.random.default_rng()
    n = 4000
    grp = rng.choice(["g1", "g2"], n, p=[0.3, 0.7])
    truth = rng.random(n) < np.where(grp == "g1", 0.3, 0.5)
    pred = np.where(rng.random(n) < 0.85, truth, ~truth)
    df = pd.DataFrame({"y": truth.astype(int), "yhat": pred.astype(int), "grp": grp})
    noisy, eps = dp_group_confusion(df, "y", "yhat", "grp", 1, ["g1", "g2"], 1.0)
    m = fairness_metrics(noisy, ["g1", "g2"], "g1", "g2", n)
    assert noisy.shape == (2, 2, 2) and 0.75 < m["accuracy"] < 0.95 and m["statistical_parity_difference"] < 0.05
    return {"epsilon": eps}
