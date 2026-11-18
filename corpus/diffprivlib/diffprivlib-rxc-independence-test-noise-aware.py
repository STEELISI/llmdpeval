import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Laplace


def chi_square(t, floor=1.0):
    """Pearson statistic of one or many r x c tables (last two axes), expected counts from the table's margins."""
    t = np.maximum(t, 0.0)
    n = t.sum(axis=(-2, -1), keepdims=True)
    e = t.sum(axis=-1, keepdims=True) * t.sum(axis=-2, keepdims=True) / np.maximum(n, floor)
    return (((t - e) ** 2) / np.maximum(e, floor)).sum(axis=(-2, -1))


def dp_rxc_independence(df, row_column, row_levels, col_column, col_levels, epsilon, n_sims=2000, alpha=0.05):
    """Noisy r x c table (PUBLIC level lists) and a chi-square test of independence whose null distribution
    includes the DP noise (parametric bootstrap). Replace-one (bounded DP, n public). Returns (dict, epsilon_spent)."""
    if 1.0 / (n_sims + 1) >= alpha:
        raise ValueError("n_sims too small: the smallest Monte Carlo p-value is 1/(n_sims+1)")
    r = df[row_column].map({v: i for i, v in enumerate(row_levels)})
    c = df[col_column].map({v: i for i, v in enumerate(col_levels)})
    keep = (r.notna() & c.notna()).to_numpy()                      # rows outside the lists are dropped
    table = np.zeros((len(row_levels), len(col_levels)))
    np.add.at(table, (r[keep].astype(int).to_numpy(), c[keep].astype(int).to_numpy()), 1)
    # replace-one: one row leaves one cell and enters another (L1 = 2).
    # Laplace mechanism at that sensitivity on every entry = ONE release; record one spend.
    acc, mech = BudgetAccountant(), Laplace(epsilon=epsilon, sensitivity=2.0)
    scale = (mech.variance(0.0) / 2) ** 0.5                          # Laplace variance = 2 scale^2
    noisy = np.vectorize(mech.randomise)(table)                      # the only access to the data
    acc.spend(epsilon, 0)
    # Post-processing: estimate the null (independence) from the noisy margins, then simulate tables
    # WITH the same noise and compare statistics.
    t = np.maximum(noisy, 0.0)
    n = max(int(round(t.sum())), 1)
    p0 = np.outer(t.sum(axis=1), t.sum(axis=0)).ravel() / max(t.sum(), 1.0) ** 2
    sims = np.random.multinomial(n, p0 / p0.sum(), size=n_sims).reshape(n_sims, *table.shape)
    sims = sims + np.random.laplace(0.0, scale, sims.shape)
    stat = chi_square(noisy)
    p_value = (1 + np.sum(chi_square(sims) >= stat)) / (n_sims + 1)
    return {"table": t, "row_margin": t.sum(axis=1), "col_margin": t.sum(axis=0), "chi2": stat,
            "p_value": p_value, "min_p": 1.0 / (n_sims + 1)}, acc.total()[0]


def _test():
    import pandas as pd
    m = 3000
    a = np.random.choice(["a1", "a2", "a3"], m)
    b = np.where(np.random.rand(m) < np.where(a == "a1", 0.5, 0.3), "b1", np.random.choice(["b2", "b3", "b4"], m))
    df = pd.DataFrame({"x": a, "y": b})
    res, eps = dp_rxc_independence(df, "x", ["a1", "a2", "a3"], "y", ["b1", "b2", "b3", "b4"], epsilon=1.0)
    assert res["table"].shape == (3, 4) and res["min_p"] <= res["p_value"] <= 1
    return {"epsilon": float(eps)}
