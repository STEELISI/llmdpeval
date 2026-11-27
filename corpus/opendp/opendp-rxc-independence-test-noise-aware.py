import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


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
    r, c = len(row_levels), len(col_levels)
    cells = [f"{a}|{b}" for a in row_levels for b in col_levels]    # public list of the r*c cells
    labels = (df[row_column].astype(str) + "|" + df[col_column].astype(str)).tolist()
    space = (dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance())
    make = lambda s: space >> dp.t.then_count_by_categories(categories=cells) >> dp.m.then_laplace(scale=s)
    scale = dp.binary_search_param(make, d_in=2, d_out=epsilon)     # replace-one = symmetric distance 2
    meas = make(scale)
    noisy = np.array(meas(labels)[:-1], dtype=float).reshape(r, c)  # one release; drop the null bin
    # Post-processing: estimate the null (independence) from the noisy margins, then simulate tables
    # WITH the same noise and compare statistics.
    t = np.maximum(noisy, 0.0)
    n = max(int(round(t.sum())), 1)
    p0 = np.outer(t.sum(axis=1), t.sum(axis=0)).ravel() / max(t.sum(), 1.0) ** 2
    sims = np.random.multinomial(n, p0 / p0.sum(), size=n_sims).reshape(n_sims, r, c)
    q = 1 - np.exp(-1 / scale)                    # discrete Laplace(scale) = difference of two geometrics
    sims = sims + np.random.geometric(q, sims.shape) - np.random.geometric(q, sims.shape)
    stat = chi_square(noisy)
    p_value = (1 + np.sum(chi_square(sims) >= stat)) / (n_sims + 1)
    return {"table": t, "row_margin": t.sum(axis=1), "col_margin": t.sum(axis=0), "chi2": stat,
            "p_value": p_value, "min_p": 1.0 / (n_sims + 1)}, meas.map(2)


def _test():
    import pandas as pd
    m = 3000
    a = np.random.choice(["a1", "a2", "a3"], m)
    b = np.where(np.random.rand(m) < np.where(a == "a1", 0.5, 0.3), "b1", np.random.choice(["b2", "b3", "b4"], m))
    df = pd.DataFrame({"x": a, "y": b})
    res, eps = dp_rxc_independence(df, "x", ["a1", "a2", "a3"], "y", ["b1", "b2", "b3", "b4"], epsilon=1.0)
    assert res["table"].shape == (3, 4) and res["min_p"] <= res["p_value"] <= 1
    return {"epsilon": float(eps)}
