import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_binned_histogram(df, column, breaks, max_length, epsilon):
    """Histogram of a numeric column over public bin edges, with the polars Context. .cut turns
    values into bin labels; the full label list is known in advance, so it is passed as public keys
    and no delta is needed. One person = one row (replace-one: bounded DP, n public).

    breaks: public, sorted bin edges. Returns ({bin label: count}, epsilon_spent).
    """
    # Labels exactly as polars' cut writes them: (-inf, b0], (b0, b1], ..., (bk, inf].
    # Let polars write them from PUBLIC values (each edge, then inf), so the text always matches.
    breaks = [float(b) for b in breaks]
    labels = pl.Series(column, [*breaks, float("inf")]).cut(breaks=breaks)
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1),   # replace-one: OpenDP doubles the add/remove noise
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query()
             .with_columns(pl.col(column).cast(float).cut(breaks=breaks))
             .group_by(column)
             .agg(dp.len(signed=True))
             .with_keys(pl.LazyFrame([labels])))   # every bin is reported, even empty ones
    out = query.release().collect()
    return dict(zip(out[column].cast(str).to_list(), out["len"].to_list())), context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 10) for i in range(200)]})
    hist, epsilon_spent = dp_binned_histogram(df, "value", [2.0, 5.0, 8.0], 1000, 1.0)
    assert len(hist) == 4
    return {"epsilon": float(epsilon_spent)}
