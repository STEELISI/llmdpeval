import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_sum_after_cleaning(df, column, fill_value, bounds, max_length, epsilon, n_is_public=True):
    """Clean a column with ordinary polars steps (cast, fill_null, clip) in with_columns, then release
    a plain .sum() with .dp.noise(). OpenDP tracks the clip bounds as a domain fact and uses them to
    set the sensitivity. One person = one row. Default: replace-one (n public),
    unit_of(changes=1). n_is_public=False: add/remove-one, unit_of(contributions=1).

    fill_value and integer bounds are public. Returns (private_sum, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1) if n_is_public else dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query()
             # integer cast avoids the extra float-rounding term in a float sum's sensitivity;
             # failed casts give null, so fill before clip (fill_null drops known bounds)
             .with_columns(pl.col(column).cast(int).fill_null(fill_value).clip(*bounds))
             .select(pl.col(column).sum().dp.noise()))
    print(query.summarize().to_dicts())
    value = query.release().collect().item()
    return value, context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [i % 11 - 5 if i % 13 else None for i in range(200)]})
    _, eps_r1 = dp_sum_after_cleaning(df, "value", 0, (-5, 10), 1000, 1.0)
    _, eps_ar = dp_sum_after_cleaning(df, "value", 0, (-5, 10), 1000, 1.0, n_is_public=False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
