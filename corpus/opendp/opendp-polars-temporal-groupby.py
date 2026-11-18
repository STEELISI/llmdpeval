import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_counts_per_year(df, date_column, date_format, years, max_length, epsilon):
    """Counts per calendar year from a text date column, with the polars Context. The text is
    parsed inside the query (str.to_date), the year is taken with dt.year, and the public list of
    years is passed as keys. One person = one row (replace-one: bounded DP, n public).

    years: public list of years to report. Returns ({year: count}, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1),   # replace-one: OpenDP doubles the add/remove noise
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query()
             .with_columns(pl.col(date_column).str.to_date(format=date_format))  # bad text -> null
             .with_columns(YEAR=pl.col(date_column).dt.year())
             .group_by("YEAR")
             .agg(dp.len(signed=True))
             .with_keys(pl.LazyFrame({"YEAR": years}, schema={"YEAR": pl.Int32})))  # dt.year is Int32
    out = query.release().collect()
    return dict(zip(out["YEAR"].to_list(), out["len"].to_list())), context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"when": [f"20{20 + i % 3}-0{1 + i % 9}-15" for i in range(150)]})
    _, epsilon_spent = dp_counts_per_year(df, "when", "%Y-%m-%d", [2020, 2021, 2022], 1000, 1.0)
    return {"epsilon": float(epsilon_spent)}
