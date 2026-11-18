import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_grouped_sum_mean(df, group_column, keys, value_column, bounds, max_length, epsilon):
    """Per-group count, sum and mean of a numeric column with the polars Context. The group keys come
    from a public list (with_keys), so no keys are learned from the data and the release is pure DP.
    One person = one row (replace-one: bounded DP, n public).

    keys: public list of group labels; bounds: public (L, U). Returns (list of row dicts, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1),   # replace-one: OpenDP doubles the add/remove noise
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query()
             .group_by(group_column)
             .agg(dp.len(signed=True),
                  pl.col(value_column).cast(float).dp.sum(bounds).alias("sum"),
                  pl.col(value_column).cast(float).dp.mean(bounds).alias("mean"))
             .with_keys(pl.LazyFrame({group_column: keys})))   # public keys: every key gets a row
    print(query.summarize().to_dicts())    # one scale per statistic; the budget is split inside
    rows = query.release().collect().sort(group_column).to_dicts()
    return rows, context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"grp": ["a", "b", "c", "a"] * 50, "value": [float(i % 10) for i in range(200)]})
    _, epsilon_spent = dp_grouped_sum_mean(df, "grp", ["a", "b", "c", "d"], "value", (0.0, 10.0), 1000, 1.0)
    return {"epsilon": float(epsilon_spent)}
