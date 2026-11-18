import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Laplace
from scipy.stats import chi2, norm


def pool(diff, var):
    """Post-processing: fixed-effect pooled estimate, Cochran Q heterogeneity test and DerSimonian-Laird tau^2.
    var must hold the noise-aware variance of each per-experiment difference."""
    w = 1.0 / var
    est = np.sum(w * diff) / np.sum(w)
    q = np.sum(w * (diff - est) ** 2)
    dfree = len(diff) - 1
    tau2 = max(0.0, (q - dfree) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    w_re = 1.0 / (var + tau2)
    est_re = np.sum(w_re * diff) / np.sum(w_re)
    se, se_re = np.sqrt(1 / np.sum(w)), np.sqrt(1 / np.sum(w_re))
    return {"fixed": est, "fixed_se": se, "fixed_p": 2 * norm.sf(abs(est / se)), "q": q,
            "q_p": chi2.sf(q, dfree), "tau2": tau2, "random": est_re, "random_se": se_re}


def dp_pooled_effect(df, exp_column, experiments, arm_column, treated_value, value_column, bounds, epsilon):
    """Treated-minus-control mean difference in each of many disjoint experiments (PUBLIC experiment list),
    pooled across them. All sizes are PRIVATE. Replace-one (bounded DP, n public). Returns (dict, epsilon_spent)."""
    lo, hi = bounds
    e = df[exp_column].map({x: i for i, x in enumerate(experiments)})
    keep = e.notna().to_numpy()
    x = ((df[value_column].astype(float).clip(lo, hi) - lo) / (hi - lo)).to_numpy()[keep]   # in [0, 1]
    arm = (df[arm_column] == treated_value).to_numpy()[keep].astype(int)                  # 1 = treated
    stats = np.zeros((len(experiments), 2, 3))                       # (n, sum, sum of squares) per cell
    np.add.at(stats, (e[keep].astype(int).to_numpy(), arm), np.column_stack([np.ones_like(x), x, x * x]))
    # one row adds (1, x, x^2), L1 <= 3, to ONE cell; replace-one can move it: L1 = 6.
    # Laplace mechanism at that sensitivity on every entry = ONE release; record one spend.
    acc, mech = BudgetAccountant(), Laplace(epsilon=epsilon, sensitivity=6.0)
    scale = (mech.variance(0.0) / 2) ** 0.5                          # Laplace variance = 2 scale^2
    n, s, q = np.moveaxis(np.vectorize(mech.randomise)(stats), -1, 0)
    acc.spend(epsilon, 0)
    n = np.maximum(n, 2.0)                                           # post-processing from here on
    m = np.clip(s / n, 0, 1)
    # One noisy sum of squares per cell is too noisy for a variance: pool it over experiments, per arm.
    sd2 = np.clip(q.sum(0) / n.sum(0) - (s.sum(0) / n.sum(0)) ** 2, 1e-6, 0.25)
    samp = sd2 / n                                                   # sampling variance of each mean
    noise = 2 * scale ** 2 * (1 + m * m) / n ** 2                    # DP noise variance (delta method)
    diff = (m[:, 1] - m[:, 0]) * (hi - lo)
    dvar = (samp + noise).sum(axis=1) * (hi - lo) ** 2
    return {"diff": dict(zip(experiments, diff)), "se": dict(zip(experiments, np.sqrt(dvar))),
            "se_sampling": dict(zip(experiments, np.sqrt(samp.sum(axis=1)) * (hi - lo))),
            **pool(diff, dvar)}, acc.total()[0]


def _test():
    import pandas as pd
    m = 12000
    exp = np.random.choice([f"e{i}" for i in range(30)], m)
    arm = np.random.choice(["t", "c"], m)
    val = np.clip(np.random.normal(np.where(arm == "t", 3.2, 3.0), 1.0), 1, 5)
    df = pd.DataFrame({"exp": exp, "arm": arm, "rating": val})
    res, eps = dp_pooled_effect(df, "exp", [f"e{i}" for i in range(30)], "arm", "t", "rating", (1.0, 5.0), 1.0)
    assert len(res["diff"]) == 30 and res["fixed_se"] > 0 and res["tau2"] >= 0
    return {"epsilon": float(eps)}
