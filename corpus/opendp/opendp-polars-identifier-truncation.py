import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_mean_with_person_ids(df, id_column, value_column, bounds, max_rows_per_person, max_length, epsilon):
    """Mean of a column when one person may own many rows, identified by an id column. The privacy
    unit is "all rows of one id"; the query must truncate each id to a public number of rows,
    which bounds the sensitivity. Replace-one person (bounded DP: the number of ids is public).

    max_rows_per_person, bounds and max_length are public. Returns (private_mean, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1, identifier=pl.col(id_column)),   # replace one person's rows
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query()
             .truncate_per_group(max_rows_per_person)     # keep at most k random rows per id
             .select(pl.col(value_column).cast(float).dp.mean(bounds)))
    print(query.summarize().to_dicts())    # noise grows with k: bias (dropped rows) vs variance
    value = query.release().collect().item()
    return value, context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"pid": [i % 40 for i in range(200)], "value": [float(i % 10) for i in range(200)]})
    _, epsilon_spent = dp_mean_with_person_ids(df, "pid", "value", (0.0, 10.0), 5, 1000, 1.0)
    return {"epsilon": float(epsilon_spent)}
