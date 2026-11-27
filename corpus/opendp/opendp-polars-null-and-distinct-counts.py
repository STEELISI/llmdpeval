import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_missingness_profile(df, column, max_length, epsilon, n_is_public=True):
    """Data-quality counts for one column: non-null count, null count and number of distinct values,
    each as a DP counting query in the polars Context. One person = one row. Default: replace-one (n public),
    unit_of(changes=1). n_is_public=False: add/remove-one, unit_of(contributions=1).
    Each count moves by at most 1 under either relation; OpenDP's changes=1 charge is conservative (2x).

    Returns ({"count", "null_count", "n_unique"}, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1) if n_is_public else dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    # signed=True: Int64 output that may go negative (OpenDP 0.16 warns if signed is omitted).
    query = context.query().select(
        pl.col(column).dp.count(signed=True).alias("count"),
        pl.col(column).dp.null_count(signed=True).alias("null_count"),
        pl.col(column).dp.n_unique(signed=True).alias("n_unique"),
    )
    print(query.summarize().to_dicts())    # the three share one budget slot
    row = query.release().collect().to_dicts()[0]
    return row, context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 7) if i % 9 else None for i in range(200)]})
    _, eps_r1 = dp_missingness_profile(df, "value", 1000, 1.0)
    _, eps_ar = dp_missingness_profile(df, "value", 1000, 1.0, n_is_public=False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
