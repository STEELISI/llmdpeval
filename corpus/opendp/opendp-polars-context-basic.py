import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_table_stats(df, column, bounds, max_length, epsilon, n_is_public=True):
    """Row count, DP sum and DP mean of one column with the polars Context.

    Default: replace-one (n public): unit_of(changes=1); OpenDP releases the length without noise
    and charges the sum U - L. n_is_public=False: add/remove-one, unit_of(contributions=1); the
    length gets noise and the sum is charged max(|L|, |U|).

    bounds and max_length (a public upper bound on the row count) must be public.
    Returns ({"count", "sum", "mean"}, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1) if n_is_public else dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=2,                                  # two releases below
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    value = pl.col(column).cast(float).fill_null((bounds[0] + bounds[1]) / 2)  # public fill value
    count_query = context.query().select(dp.len(signed=True))
    dp_count = count_query.release().collect().item()
    # Sum and length in one slot, divided afterwards (under changes=1 the length is exact n).
    sum_query = context.query().select(value.dp.sum(bounds=bounds), dp.len(signed=True).alias("n"))
    print(sum_query.summarize(alpha=0.05).to_dicts())       # scales and accuracy before releasing
    row = sum_query.release().collect()
    dp_sum, dp_n = row[column].item(), row["n"].item()
    result = {"count": dp_count, "sum": dp_sum, "mean": dp_sum / max(dp_n, 1)}
    epsilon_spent = context.accountant.map(context.d_in)
    return result, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, eps_r1 = dp_table_stats(df, "value", (-5.0, 5.0), max_length=1000, epsilon=1.0)
    _, eps_ar = dp_table_stats(df, "value", (-5.0, 5.0), 1000, 1.0, n_is_public=False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
