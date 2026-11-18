import numpy as np


def dp_contingency_tables(df, pairs, categories, epsilon, relation="replace_one"):
    """eps-DP two-way tables for several column pairs; categories: {column: PUBLIC list}.

    One record touches one cell in EACH of the m tables, so the stacked L1 sensitivity is m
    (add/remove) or 2m (replace-one, bounded DP with public n; the default). Returns ({(col_a, col_b): 2-D array}, epsilon_spent).
    """
    rng = np.random.default_rng()
    m = len(pairs)
    l1 = m * (1.0 if relation == "add_remove" else 2.0)
    scale = l1 / epsilon
    out = {}
    for a, b in pairs:
        ia = {v: i for i, v in enumerate(categories[a])}
        ib = {v: i for i, v in enumerate(categories[b])}
        table = np.zeros((len(ia), len(ib)))
        for va, vb in zip(df[a].to_numpy(), df[b].to_numpy()):
            if va in ia and vb in ib:                 # values outside the public lists are dropped
                table[ia[va], ib[vb]] += 1
        out[(a, b)] = table + rng.laplace(0.0, scale, size=table.shape)
    epsilon_spent = l1 / scale
    assert epsilon_spent <= epsilon + 1e-9
    return out, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"p": ["x", "y", "x", "z"] * 30, "q": ["u", "v", "v", "u"] * 30,
                       "r": ["s", "t", "s", "s"] * 30})
    cats = {"p": ["x", "y", "z"], "q": ["u", "v"], "r": ["s", "t"]}
    res, eps = dp_contingency_tables(df, [("p", "q"), ("q", "r")], cats, epsilon=1.0)
    assert res[("p", "q")].shape == (3, 2)
    return {"epsilon": eps}
