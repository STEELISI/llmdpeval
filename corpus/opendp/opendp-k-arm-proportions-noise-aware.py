from statistics import NormalDist

import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


def dp_k_arm_proportions(df, arm_column, arms, outcome_column, outcome_value, epsilon, alpha=0.05):
    """Outcome rate in each of k arms (PUBLIC arm list, arms[0] = control) and every arm-vs-control difference with
    a noise-aware SE, CI and Holm-adjusted p-value. Arm sizes are PRIVATE. Replace-one (bounded DP, n public).
    Returns (dict, epsilon_spent)."""
    cells = [f"{a}|{y}" for a in arms for y in (1, 0)]              # public list: arm x outcome yes/no
    labels = (df[arm_column].astype(str) + "|" + (df[outcome_column] == outcome_value).astype(int).astype(str))
    space = (dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance())
    make = lambda s: space >> dp.t.then_count_by_categories(categories=cells) >> dp.m.then_laplace(scale=s)
    # replace-one = symmetric distance 2 (one row leaves one cell and enters another)
    scale = dp.binary_search_param(make, d_in=2, d_out=epsilon)
    meas = make(scale)
    counts = np.array(meas(labels.tolist())[:-1], dtype=float)     # one release; drop the null bin
    yes, no = np.maximum(counts.reshape(len(arms), 2), 0.0).T      # clamp (post-processing)
    v = 2 * scale ** 2                                              # discrete Laplace variance, about 2 scale^2
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
    return res, meas.map(2)


def _test():
    import pandas as pd
    m = 4000
    arm = np.random.choice(["c", "t1", "t2", "t3"], m)
    out = np.random.rand(m) < np.where(arm == "t3", 0.45, 0.30)
    df = pd.DataFrame({"arm": arm, "out": np.where(out, "yes", "no")})
    res, eps = dp_k_arm_proportions(df, "arm", ["c", "t1", "t2", "t3"], "out", "yes", epsilon=1.0)
    assert set(res["contrasts"]) == {"t1", "t2", "t3"} and all(0 <= c["p_holm"] <= 1 for c in res["contrasts"].values())
    return {"epsilon": float(eps)}
