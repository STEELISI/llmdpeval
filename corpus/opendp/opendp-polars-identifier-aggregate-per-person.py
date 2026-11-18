import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_mean_of_person_means(df, id_column, value_column, bounds, max_length, epsilon, n_is_public=True):
    """Mean over people (not rows) when one person owns many rows: first collapse each id to one
    row with an ordinary group_by(id).agg(...), then take the DP mean. Collapsing to one row per id
    caps each person's influence, so no truncation bias and no extra noise.
    Default: replace-one person (number of people public): unit_of(changes=1, identifier=id).
    n_is_public=False: add/remove all rows of one id: unit_of(contributions=1, identifier=id).

    bounds are public. Returns (private_mean, epsilon_spent).
    """
    unit = (dp.unit_of(changes=1, identifier=pl.col(id_column)) if n_is_public
            else dp.unit_of(contributions=1, identifier=pl.col(id_column)))
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=unit,
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = (context.query()
             .group_by(pl.col(id_column))                 # one row per person (truncation step)
             .agg(pl.col(value_column).mean())   # any per-person summary works; mean is float
             .select(pl.col(value_column).dp.mean(bounds)))
    print(query.summarize().to_dicts())
    value = query.release().collect().item()
    return value, context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"pid": [i % 20 for i in range(60)], "value": [float(i % 10) for i in range(60)]})
    _, eps_r1 = dp_mean_of_person_means(df, "pid", "value", (-5.0, 10.0), 1000, 1.0)
    _, eps_ar = dp_mean_of_person_means(df, "pid", "value", (-5.0, 10.0), 1000, 1.0, n_is_public=False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
