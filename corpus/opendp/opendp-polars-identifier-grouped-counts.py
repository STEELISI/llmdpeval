import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_group_counts_with_person_ids(df, id_column, group_column, keys, max_rows_per_group,
                                    max_length, epsilon):
    """Counts per group when one person (an id) may have many rows in many groups. Truncate each id
    to a public number of rows per group; with public keys, OpenDP bounds how many groups one id can
    touch by the key count. Neighbours: replace-one person (all rows of one id change).

    keys: public group labels. Returns ({key: count}, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1, identifier=pl.col(id_column)),   # replace one person's rows
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query()
             .truncate_per_group(max_rows_per_group, by=[group_column])  # <= k rows per id per group
             .group_by(group_column)
             .agg(dp.len(signed=True))
             .with_keys(pl.LazyFrame({group_column: keys})))
    print(query.summarize().to_dicts())    # scale grows with k * (number of keys)
    out = query.release().collect()
    return dict(zip(out[group_column].to_list(), out["len"].to_list())), context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"pid": [i % 50 for i in range(300)], "grp": ["a", "b", "c"] * 100})
    _, epsilon_spent = dp_group_counts_with_person_ids(df, "pid", "grp", ["a", "b", "c"], 1, 1000, 1.0)
    return {"epsilon": float(epsilon_spent)}
