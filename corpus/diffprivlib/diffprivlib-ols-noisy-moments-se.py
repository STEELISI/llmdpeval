import math

import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Laplace


def dp_ols_with_se(df, target, target_bounds, numeric, categorical, epsilon, draws=300):
    """OLS with several regressors (numeric + one-hot of PUBLIC levels) from one noisy moment vector,
    with noise-aware standard errors and p-values. Replace-one neighbours (bounded DP, n public).

    numeric: {column: (low, high)} PUBLIC bounds; categorical: {column: [PUBLIC levels]}, first = baseline.
    Returns ({name: {"coef", "se", "p"}} in original units, epsilon_spent).
    """
    cols, names, scale_back = [np.ones(len(df))], [], []
    for c, (lo, hi) in numeric.items():          # min-max with PUBLIC bounds: every entry in [0, 1]
        cols.append((df[c].astype(float).clip(lo, hi) - lo) / (hi - lo)); names.append(c); scale_back.append(hi - lo)
    for c, levels in categorical.items():        # public levels: same columns whatever the data holds
        for lv in levels[1:]:
            cols.append((df[c] == lv).astype(float).to_numpy()); names.append(f"{c}={lv}"); scale_back.append(1.0)
    ly, uy = target_bounds
    Z = np.column_stack(cols + [(df[target].astype(float).clip(ly, uy) - ly) / (uy - ly)])
    iu = np.triu_indices(Z.shape[1])
    # One row: each block (intercept, numeric column, one-hot group, target) adds <= 1 to sum|z| and to
    # sum z^2, so sum_{i<=j} |z_i z_j| <= (R^2 + R) / 2 with R blocks. Replace-one = remove + add: x2.
    R = 2 + len(numeric) + len(categorical)
    sens = float(R * R + R)
    acc = BudgetAccountant()
    mech = Laplace(epsilon=epsilon, sensitivity=sens)   # L1 vector release = this noise on every entry
    noisy = np.array([mech.randomise(v) for v in (Z.T @ Z)[iu]])
    acc.spend(epsilon, 0)                        # one release of the whole vector
    b = np.sqrt(mech.variance(0.0) / 2)          # Laplace scale from the mechanism's variance
    M = np.zeros((Z.shape[1],) * 2); M[iu] = noisy; M = M + np.triu(M, 1).T
    p = Z.shape[1] - 1                           # post-processing from here on
    solve = lambda Mx: np.linalg.solve(Mx[:p, :p], Mx[:p, p])
    beta = solve(M)
    n = len(df)                                  # public under replace-one
    sigma2 = max((M[p, p] - beta @ M[:p, p]) / (n - p), 1e-12)
    cov_sampling = sigma2 * np.linalg.inv(M[:p, :p])
    sims = []                                    # DP noise variance: redo the release's noise on M
    for _ in range(draws):
        E = np.zeros_like(M); E[iu] = np.random.laplace(0.0, b, len(iu[0])); E = E + np.triu(E, 1).T
        sims.append(solve(M + E))
    se = np.sqrt(np.diag(cov_sampling) + np.var(sims, axis=0))
    k = (uy - ly) / np.array([1.0] + scale_back)   # back to original units (intercept left scaled)
    out = {nm: {"coef": beta[i] * k[i], "se": se[i] * k[i],
                "p": math.erfc(abs(beta[i] / se[i]) / math.sqrt(2))} for i, nm in enumerate(names, 1)}
    return out, acc.total()[0]


def _test():
    import pandas as pd
    m = 4000
    x = np.random.uniform(0, 50, m)
    g = np.random.choice(["u", "v", "w"], m)
    y = np.clip(10 + 0.5 * x + np.where(g == "v", 8.0, 0.0) + np.random.normal(0, 3, m), 0, 60)
    df = pd.DataFrame({"x": x, "g": g, "y": y})
    res, eps = dp_ols_with_se(df, "y", (0, 60), {"x": (0, 50)}, {"g": ["u", "v", "w"]}, epsilon=1.0)
    assert set(res) == {"x", "g=v", "g=w"} and all(r["se"] > 0 for r in res.values())
    return {"epsilon": float(eps)}
