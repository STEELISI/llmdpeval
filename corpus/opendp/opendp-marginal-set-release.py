import itertools

import numpy as np
import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_marginal_set(df, marginals, categories, max_length, epsilon):
    """1-way and 2-way marginals over PUBLIC category lists under one epsilon (replace-one: bounded DP, n public).

    marginals: list of column tuples, e.g. [("a",), ("b",), ("a", "b")]; categories: {column: public list}.
    A changed row moves between two cells of EVERY table, so the Context splits epsilon evenly over the m tables
    and charges each its replace-one cost. Returns ({tuple: noisy ndarray}, epsilon_spent).
    """
    cols = sorted({c for m in marginals for c in m})
    data = pl.from_pandas(df[cols].astype(str)).lazy()
    context = dp.Context.compositor(
        data=data,
        privacy_unit=dp.unit_of(changes=1),      # replace-one
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=len(marginals),        # one query per table
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    out = {}
    for m in marginals:
        cells = list(itertools.product(*[categories[c] for c in m]))   # public cross product, zeros included
        keys = pl.LazyFrame({c: [cell[i] for cell in cells] for i, c in enumerate(m)},
                            schema={c: pl.String for c in m})
        res = (context.query().group_by(list(m)).agg(dp.len(signed=True).alias("n"))
               .with_keys(keys).release().collect())
        lookup = {tuple(r[:-1]): r[-1] for r in res.select([*m, "n"]).iter_rows()}   # row order is not key order
        out[m] = np.array([lookup[cell] for cell in cells], dtype=float).reshape([len(categories[c]) for c in m])
    epsilon_spent = context.accountant.map(context.d_in)
    assert epsilon_spent <= epsilon + 1e-9
    return out, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"a": ["x", "y", "x", "z"] * 50, "b": ["u", "v", "v", "u"] * 50})
    cats = {"a": ["x", "y", "z", "w"], "b": ["u", "v"]}
    res, eps = dp_marginal_set(df, [("a",), ("b",), ("a", "b")], cats, 1000, epsilon=1.0)
    assert res[("a", "b")].shape == (4, 2) and res[("a",)].shape == (4,)
    return {"epsilon": float(eps)}
