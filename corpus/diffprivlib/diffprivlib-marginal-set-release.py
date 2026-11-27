import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import histogram


def dp_marginal_set(df, marginals, categories, epsilon):
    """1-way and 2-way marginals over PUBLIC category lists under one epsilon (replace-one: bounded DP, n public).

    marginals: list of column tuples, e.g. [("a",), ("b",), ("a", "b")]; categories: {column: public list}.
    Each table is one diffprivlib histogram over integer cell codes. Returns ({tuple: noisy ndarray}, epsilon_spent).
    """
    acc = BudgetAccountant()
    eps_table = epsilon / len(marginals)          # a record sits in one cell of EVERY table: split evenly
    out = {}
    for m in marginals:
        shape = [len(categories[c]) for c in m]
        idx = [df[c].map({v: i for i, v in enumerate(categories[c])}) for c in m]
        keep = np.all([i.notna().to_numpy() for i in idx], axis=0)   # values outside the public lists are dropped
        codes = np.ravel_multi_index([i.to_numpy()[keep].astype(int) for i in idx], shape)
        k = int(np.prod(shape))
        # histogram is calibrated for add/remove (one cell moves by 1). Replace-one moves two cells,
        # so pass half the table's share; the real guarantee is 2x what the accountant records.
        counts, _ = histogram(codes, epsilon=eps_table / 2, bins=k, range=(-0.5, k - 0.5), accountant=acc)
        out[m] = counts.astype(float).reshape(shape)
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return out, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"a": ["x", "y", "x", "z"] * 50, "b": ["u", "v", "v", "u"] * 50})
    cats = {"a": ["x", "y", "z", "w"], "b": ["u", "v"]}
    res, eps = dp_marginal_set(df, [("a",), ("b",), ("a", "b")], cats, epsilon=1.0)
    assert res[("a", "b")].shape == (4, 2) and res[("a",)].shape == (4,)
    return {"epsilon": float(eps)}
