import numpy as np
from scipy.stats import norm


def _fixed_q(diff, var):
    """Fixed-effect estimate and Cochran Q over the last axis (vectorised for the bootstrap)."""
    w = 1.0 / var
    est = np.sum(w * diff, -1) / np.sum(w, -1)
    return est, np.sum(w * (diff - est[..., None]) ** 2, -1)


def _diffs(n, s, sd2, scale, width):
    """Post-processing of noisy (count, sum) cells: per-experiment difference and noise-aware variance."""
    n = np.maximum(n, 2.0)
    m = np.clip(s / n, 0, 1)
    var = sd2 / n + 2 * scale ** 2 * (1 + m * m) / n ** 2           # sampling + DP noise (delta method)
    return (m[..., 1] - m[..., 0]) * width, var.sum(-1) * width ** 2, m


def dp_pooled_effect(df, exp_column, experiments, arm_column, treated_value, value_column, bounds, epsilon,
                     n_boot=999):
    """Treated-minus-control mean difference in each of many disjoint experiments (PUBLIC experiment list),
    pooled across them. All sizes are PRIVATE. Replace-one (bounded DP, n public). Returns (dict, epsilon_spent)."""
    rng = np.random.default_rng()
    lo, hi = bounds
    e = df[exp_column].map({x: i for i, x in enumerate(experiments)})
    keep = e.notna().to_numpy()
    x = ((df[value_column].astype(float).clip(lo, hi) - lo) / (hi - lo)).to_numpy()[keep]   # in [0, 1]
    arm = (df[arm_column] == treated_value).to_numpy()[keep].astype(int)                  # 1 = treated
    stats = np.zeros((len(experiments), 2, 3))                       # (n, sum, sum of squares) per cell
    np.add.at(stats, (e[keep].astype(int).to_numpy(), arm), np.column_stack([np.ones_like(x), x, x * x]))
    sens = 6.0       # one row adds (1, x, x^2), L1 <= 3, to ONE cell; replace-one can move it: L1 = 6
    scale = sens / epsilon
    n, s, q = np.moveaxis(stats + rng.laplace(0.0, scale, stats.shape), -1, 0)   # one release
    nc = np.maximum(n, 2.0)                                          # post-processing from here on
    # One noisy sum of squares per cell is too noisy for a variance: pool it over experiments, per arm.
    sd2 = np.clip(q.sum(0) / nc.sum(0) - (s.sum(0) / nc.sum(0)) ** 2, 1e-6, 0.25)
    diff, dvar, m = _diffs(n, s, sd2, scale, hi - lo)
    est, qstat = _fixed_q(diff, dvar)
    # Q p-value by parametric bootstrap with the SAME Laplace noise (heavy tails: a chi-square reference
    # over-rejects). Null: one common difference; released cell sizes and pooled control mean as the truth.
    mu_c = np.clip(s[:, 0].sum() / nc[:, 0].sum(), 0, 1)
    mu = np.array([mu_c, np.clip(mu_c + est / (hi - lo), 0, 1)])
    f = np.sqrt(np.clip(1 - 2 * scale ** 2 / np.maximum(nc.var(0), 1e-12), 0, 1))   # remove noise spread
    nc = np.maximum(nc.mean(0) + f * (nc - nc.mean(0)), 2.0)        # from the sizes used as the truth
    shape = (n_boot,) + nc.shape
    s_b = nc * mu + rng.normal(0.0, 1.0, shape) * np.sqrt(nc * sd2) + rng.laplace(0.0, scale, shape)
    _, q_b = _fixed_q(*_diffs(nc + rng.laplace(0.0, scale, shape), s_b, sd2, scale, hi - lo)[:2])
    w = 1.0 / dvar
    tau2 = max(0.0, (qstat - (len(diff) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))   # DerSimonian-Laird
    est_re, _ = _fixed_q(diff, dvar + tau2)
    se, se_re = np.sqrt(1 / w.sum()), np.sqrt(1 / np.sum(1 / (dvar + tau2)))
    return {"diff": dict(zip(experiments, diff)), "se": dict(zip(experiments, np.sqrt(dvar))),
            "fixed": est, "fixed_se": se, "fixed_p": 2 * norm.sf(abs(est / se)), "q": qstat,
            "q_p": (1 + np.sum(q_b >= qstat)) / (n_boot + 1), "tau2": tau2, "random": est_re,
            "random_se": se_re}, sens / scale


def _test():
    import pandas as pd
    rng = np.random.default_rng()
    m = 12000
    exp = rng.choice([f"e{i}" for i in range(30)], m)
    arm = rng.choice(["t", "c"], m)
    val = np.clip(rng.normal(np.where(arm == "t", 3.2, 3.0), 1.0), 1, 5)
    df = pd.DataFrame({"exp": exp, "arm": arm, "rating": val})
    res, eps = dp_pooled_effect(df, "exp", [f"e{i}" for i in range(30)], "arm", "t", "rating", (1.0, 5.0), 1.0)
    assert len(res["diff"]) == 30 and res["fixed_se"] > 0 and res["tau2"] >= 0 and 0 < res["q_p"] <= 1
    return {"epsilon": float(eps)}
