import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_count_per_group(df, group_column, keys, max_length, epsilon):
    """DP count per group with a PUBLIC key list (pure DP, replace-one neighbours:
    bounded DP, n public).

    Keys absent from the data still get a noisy count, so absence does not leak.
    Returns ({key: count}, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1),   # replace-one: one changed row (OpenDP doubles the noise)
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    public_keys = pl.LazyFrame({group_column: keys}, schema={group_column: pl.String})
    query = (context.query()
             .group_by(group_column)
             .agg(dp.len(signed=True).alias("count"))
             .with_keys(public_keys))
    out = query.release().collect()
    counts = dict(zip(out[group_column].to_list(), out["count"].to_list()))   # row order is not key order
    epsilon_spent = context.accountant.map(context.d_in)
    return {k: counts[k] for k in keys}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c"] * 25})
    counts, epsilon_spent = dp_count_per_group(df, "group", ["a", "b", "c", "d"], 1000, epsilon=1.0)
    assert list(counts) == ["a", "b", "c", "d"]
    return {"epsilon": float(epsilon_spent)}
