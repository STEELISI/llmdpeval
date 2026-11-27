from statistics import NormalDist

import numpy as np


def dp_k_arm_proportions(df, arm_column, arms, outcome_column, outcome_value, epsilon, alpha=0.05):
    """Outcome rate in each of k arms (PUBLIC arm list, arms[0] = control) and every arm-vs-control difference with
    a noise-aware SE, CI and Holm-adjusted p-value. Arm sizes are PRIVATE. Replace-one (bounded DP, n public).
    Returns (dict, epsilon_spent)."""
    rng = np.random.default_rng()
    k = len(arms)
    arm = df[arm_column].map({a: i for i, a in enumerate(arms)})
    keep = arm.notna().to_numpy()                                   # rows outside the public list are dropped
    y = (df[outcome_column] == outcome_value).to_numpy()[keep]
    table = np.zeros((k, 2))                                        # columns: outcome yes, outcome no
    np.add.at(table, (arm[keep].astype(int).to_numpy(), np.where(y, 0, 1)), 1)
    sens = 2.0                         # replace-one: one row leaves one cell and enters another (L1 = 2)
    scale = sens / epsilon
    yes, no = np.maximum(table + rng.laplace(0.0, scale, table.shape), 0.0).T   # one release, then clamp
    v = 2 * scale ** 2                                              # variance of one Laplace draw
    n = np.maximum(yes + no, 2.0)                                   # noisy arm sizes, public floor
    p = yes / n
    var = p * (1 - p) / n + v * (yes ** 2 + no ** 2) / n ** 4       # sampling + noise (delta method)
    z_crit = NormalDist().inv_cdf(1 - alpha / 2)
    diff = p[1:] - p[0]
    se = np.sqrt(var[1:] + var[0])
    raw = np.array([2 * NormalDist().cdf(-abs(d / s)) for d, s in zip(diff, se)])
    order = np.argsort(raw)                                         # Holm step-down adjustment
    holm = np.minimum(1, np.maximum.accumulate(raw[order] * (len(raw) - np.arange(len(raw)))))
    adj = np.empty_like(raw)
    adj[order] = holm
    res = {"rate": dict(zip(arms, p)), "noisy_size": dict(zip(arms, n)), "contrasts": {
        a: {"diff": d, "se": s, "ci": (d - z_crit * s, d + z_crit * s), "p": r, "p_holm": h}
        for a, d, s, r, h in zip(arms[1:], diff, se, raw, adj)}}
    return res, sens / scale


def _test():
    rng = np.random.default_rng()
    import pandas as pd
    m = 4000
    arm = rng.choice(["c", "t1", "t2", "t3"], m)
    out = rng.random(m) < np.where(arm == "t3", 0.45, 0.30)
    df = pd.DataFrame({"arm": arm, "out": np.where(out, "yes", "no")})
    res, eps = dp_k_arm_proportions(df, "arm", ["c", "t1", "t2", "t3"], "out", "yes", epsilon=1.0)
    assert set(res["contrasts"]) == {"t1", "t2", "t3"} and all(0 <= c["p_holm"] <= 1 for c in res["contrasts"].values())
    return {"epsilon": float(eps)}
