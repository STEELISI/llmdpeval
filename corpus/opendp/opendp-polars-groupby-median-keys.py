import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_median_per_group(df, group_column, keys, value_column, candidates, max_length, epsilon):
    """Median of a numeric column within each group, with the polars Context. Each group's median is
    chosen from a public candidate grid by a noisy score (exponential-mechanism style); the group
    keys are public. One person = one row (replace-one: bounded DP, n public), so groups are disjoint.

    candidates: public sorted grid of possible answers. Returns ({key: median}, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1),   # replace-one: OpenDP charges more than add/remove (read summarize)
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query()
             .group_by(group_column)
             .agg(pl.col(value_column).cast(float).dp.median(candidates))
             .with_keys(pl.LazyFrame({group_column: keys})))
    print(query.summarize().to_dicts())    # distribution "ExponentialMin"
    out = query.release().collect()
    return dict(zip(out[group_column].to_list(), out[value_column].to_list())), context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"grp": ["a", "b", "c"] * 70, "value": [float(i % 10) for i in range(210)]})
    grid = [float(c) for c in range(0, 11)]
    _, epsilon_spent = dp_median_per_group(df, "grp", ["a", "b", "c"], "value", grid, 1000, 1.0)
    return {"epsilon": float(epsilon_spent)}
