from statistics import NormalDist

import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Laplace


def dp_k_arm_proportions(df, arm_column, arms, outcome_column, outcome_value, epsilon, alpha=0.05):
    """Outcome rate in each of k arms (PUBLIC arm list, arms[0] = control) and every arm-vs-control difference with
    a noise-aware SE, CI and Holm-adjusted p-value. Arm sizes are PRIVATE. Replace-one (bounded DP, n public).
    Returns (dict, epsilon_spent)."""
    k = len(arms)
    arm = df[arm_column].map({a: i for i, a in enumerate(arms)})
    keep = arm.notna().to_numpy()                                   # rows outside the public list are dropped
    y = (df[outcome_column] == outcome_value).to_numpy()[keep]
    table = np.zeros((k, 2))                                        # columns: outcome yes, outcome no
    np.add.at(table, (arm[keep].astype(int).to_numpy(), np.where(y, 0, 1)), 1)
    # replace-one: one row leaves one cell and enters another (L1 = 2). tools.histogram would
    # truncate noise at 0 and assume add/remove; the mechanism keeps the noise variance known for the test.
    # Laplace mechanism at that sensitivity on every entry = ONE release; record one spend.
    acc, mech = BudgetAccountant(epsilon=epsilon, delta=0), Laplace(epsilon=epsilon, sensitivity=2.0)
    acc.check(epsilon, 0)
    scale = (mech.variance(0.0) / 2) ** 0.5                          # Laplace variance = 2 scale^2
    yes, no = np.vectorize(mech.randomise)(table).T                 # unclamped: the noise variance stays 2 scale^2
    acc.spend(epsilon, 0)
    v = 2 * scale ** 2                                              # variance of one Laplace draw
    n = np.maximum(yes + no, 2.0)                                   # noisy arm sizes, public floor
    p = yes / n                                                     # test on unclamped counts
    pc = np.clip(p, 0, 1)
    var = pc * (1 - pc) / n + v * (yes ** 2 + no ** 2) / n ** 4     # sampling + noise (delta method)
    z_crit = NormalDist().inv_cdf(1 - alpha / 2)
    diff = p[1:] - p[0]
    se = np.sqrt(var[1:] + var[0])
    raw = np.array([2 * NormalDist().cdf(-abs(d / s)) for d, s in zip(diff, se)])
    order = np.argsort(raw)                                         # Holm step-down adjustment
    holm = np.minimum(1, np.maximum.accumulate(raw[order] * (len(raw) - np.arange(len(raw)))))
    adj = np.empty_like(raw)
    adj[order] = holm
    res = {"rate": dict(zip(arms, pc)),                             # clipped for display only
           "noisy_size": dict(zip(arms, n)), "contrasts": {
        a: {"diff": d, "se": s, "ci": (d - z_crit * s, d + z_crit * s), "p": r, "p_holm": h}
        for a, d, s, r, h in zip(arms[1:], diff, se, raw, adj)}}
    epsilon_spent = acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return res, epsilon_spent


def _test():
    import pandas as pd
    m = 4000
    arm = np.random.choice(["c", "t1", "t2", "t3"], m)
    out = np.random.rand(m) < np.where(arm == "t3", 0.45, 0.30)
    df = pd.DataFrame({"arm": arm, "out": np.where(out, "yes", "no")})
    res, eps = dp_k_arm_proportions(df, "arm", ["c", "t1", "t2", "t3"], "out", "yes", epsilon=1.0)
    assert set(res["contrasts"]) == {"t1", "t2", "t3"} and all(0 <= c["p_holm"] <= 1 for c in res["contrasts"].values())
    return {"epsilon": float(eps)}
