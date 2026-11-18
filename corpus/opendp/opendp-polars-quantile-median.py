import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_quantiles_polars(df, column, candidates, alphas, max_length, epsilon):
    """DP quantiles of one column with the polars Context: OpenDP scores each public candidate
    and picks one with noise. All quantiles share one release.
    Replace-one neighbours (bounded DP, n public): privacy_unit=unit_of(changes=1).

    candidates: a public grid of possible answers. Returns ({alpha: value}, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1),       # replace-one; OpenDP widens the noise for it
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = context.query().select(
        pl.col(column).cast(float).dp.quantile(a, candidates).alias(f"q{i}") for i, a in enumerate(alphas)
    )
    print(query.summarize().to_dicts())          # one row per quantile, distribution ExponentialMin
    row = query.release().collect()
    result = {a: row[f"q{i}"].item() for i, a in enumerate(alphas)}
    epsilon_spent = context.accountant.map(context.d_in)
    assert epsilon_spent <= epsilon + 1e-9
    return result, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 50) for i in range(200)]})
    grid = [float(c) for c in range(0, 51)]
    _, epsilon_spent = dp_quantiles_polars(df, "value", grid, [0.25, 0.5, 0.75], 1000, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
