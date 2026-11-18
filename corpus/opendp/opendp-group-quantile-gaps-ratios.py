import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_group_quantile_gaps(df, group_column, groups, value_column, candidates, alphas, group_a, group_b,
                           max_length, epsilon):
    """DP quantiles (e.g. alphas 0.1, 0.5, 0.9) per group of a PUBLIC group list, the gap group_a - group_b at
    each alpha, and each group's ratio of its highest to its lowest quantile (e.g. p90/p10).
    Replace-one neighbours (bounded DP, n public): a changed row can leave one group and join another.

    candidates: PUBLIC sorted grid of possible answers (a log-spaced grid suits ratios).
    Returns ({"quantiles", "gaps", "ratios"}, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df[[group_column, value_column]]).lazy(),
        privacy_unit=dp.unit_of(changes=1),       # replace-one; OpenDP charges both groups a row can touch
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query().group_by(group_column)
             .agg(pl.col(value_column).cast(float).dp.quantile(a, candidates).alias(f"q{i}") for i, a in enumerate(alphas))
             .with_keys(pl.LazyFrame({group_column: list(groups)})))
    out = query.release().collect()
    epsilon_spent = context.accountant.map(context.d_in)
    assert epsilon_spent <= epsilon + 1e-9
    # Post-processing only from here: sort within a group (monotone), then gaps and ratios.
    quantiles = {}
    for row in out.iter_rows(named=True):
        qs = sorted(row[f"q{i}"] for i in range(len(alphas)))
        quantiles[row[group_column]] = dict(zip(sorted(alphas), qs))
    lo, hi = min(alphas), max(alphas)
    gaps = {a: quantiles[group_a][a] - quantiles[group_b][a] for a in alphas}
    ratios = {g: (q[hi] / q[lo] if q[lo] > 0 else None) for g, q in quantiles.items()}
    return {"quantiles": quantiles, "gaps": gaps, "ratios": ratios}, epsilon_spent


def _test():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng()
    df = pd.DataFrame({"grp": rng.choice(["a", "b"], 2000), "value": rng.lognormal(3.0, 0.5, 2000)})
    grid = [float(c) for c in np.geomspace(1.0, 200.0, 120)]
    res, eps = dp_group_quantile_gaps(df, "grp", ["a", "b"], "value", grid, [0.1, 0.5, 0.9], "a", "b", 5000, 1.0)
    assert set(res["gaps"]) == {0.1, 0.5, 0.9}
    return {"epsilon": float(eps)}
