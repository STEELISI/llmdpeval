import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_counts_per_period(df, period_column, periods, max_rows_total, max_rows_per_period,
                         max_length, epsilon, n_is_public=True):
    """Counts per time period when a person appears in many periods but at most a few times in each.
    Telling the privacy unit both facts (dp.polars.Bound) lowers the sensitivity of grouped queries.
    Default: replace-one person (number of people public): one person's rows are removed and
    another's added, so both bounds double. n_is_public=False: add/remove one person (bounds as given).

    periods: public list of period labels. Returns (list of row dicts, epsilon_spent).
    """
    k = 2 if n_is_public else 1                # Bound units exist only as contributions; x2 = replace-one
    unit = dp.unit_of(contributions=[
        dp.polars.Bound(per_group=k * max_rows_total),                         # rows per person overall
        dp.polars.Bound(by=[period_column], per_group=k * max_rows_per_period),  # per person per period
    ])
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=unit,
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query()
             .group_by(period_column)
             .agg(dp.len(signed=True))
             .with_keys(pl.LazyFrame({period_column: periods})))
    print(query.summarize().to_dicts())    # under rho (Gaussian) the per-period bound helps even more
    rows = query.release().collect().sort(period_column).to_dicts()
    return rows, context.accountant.map(context.d_in)   # d_in here is the list of Bounds, not 1


def _test():
    import pandas as pd
    df = pd.DataFrame({"period": [2020 + i % 4 for i in range(240)], "pid": [i % 60 for i in range(240)]})
    _, eps_r1 = dp_counts_per_period(df, "period", [2020, 2021, 2022, 2023], 4, 1, 1000, 1.0)
    _, eps_ar = dp_counts_per_period(df, "period", [2020, 2021, 2022, 2023], 4, 1, 1000, 1.0, False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
