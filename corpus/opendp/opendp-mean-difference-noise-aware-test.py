import math

import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_mean_difference_test(df, group_column, group_a, group_b, value_column, bounds, max_length, epsilon):
    """Difference of two group means with a noise-aware z-test, polars Context. Group sizes are PRIVATE
    (noisy len per group). Replace-one neighbours (bounded DP, n public): unit_of(changes=1).

    bounds: PUBLIC (low, high). Returns ({"mean_a", "mean_b", "diff", "se", "p"}, epsilon_spent).
    """
    lo, hi = bounds
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1),          # a changed row may move between the two groups
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    v = (pl.col(value_column).cast(float).clip(lo, hi) - lo) / (hi - lo)   # public bounds -> [0, 1]
    query = (context.query()
             .group_by(group_column)
             .agg(dp.len(signed=True).alias("n"), v.dp.sum((0.0, 1.0)).alias("s"),
                  (v * v).dp.sum((0.0, 1.0)).alias("q"))
             .with_keys(pl.LazyFrame({group_column: [group_a, group_b]})))
    scales = {r["column"]: r["scale"] for r in query.summarize().to_dicts()}   # noise scales from OpenDP
    rows = {r[group_column]: r for r in query.release().collect().to_dicts()}
    out, var = {}, 0.0                                     # post-processing on released values only
    for name, g in (("a", group_a), ("b", group_b)):
        n = max(rows[g]["n"], 2.0)                         # public floor for tiny noisy sizes
        m = min(max(rows[g]["s"] / n, 0.0), 1.0)
        sample_var = min(max(rows[g]["q"] / n - m * m, 0.0), 0.25) / n
        dp_var = 2 * (scales["s"] ** 2 + m * m * scales["n"] ** 2) / n ** 2   # Laplace var = 2 scale^2
        var += sample_var + dp_var
        out[f"mean_{name}"] = lo + m * (hi - lo)
    out["diff"] = out["mean_a"] - out["mean_b"]
    out["se"] = math.sqrt(var) * (hi - lo)
    out["p"] = math.erfc(abs(out["diff"] / out["se"]) / math.sqrt(2))
    return out, context.accountant.map(context.d_in)


def _test():
    import numpy as np
    import pandas as pd
    m = 3000
    g = np.random.choice(["p", "q", "r"], m, p=[0.1, 0.6, 0.3])
    df = pd.DataFrame({"grp": g, "score": np.clip(np.random.normal(np.where(g == "p", 3.5, 3.0), 1.0), 1, 5)})
    res, eps = dp_mean_difference_test(df, "grp", "p", "q", "score", (1.0, 5.0), 10000, epsilon=1.0)
    assert res["se"] > 0 and 0 <= res["p"] <= 1
    return {"epsilon": float(eps)}
