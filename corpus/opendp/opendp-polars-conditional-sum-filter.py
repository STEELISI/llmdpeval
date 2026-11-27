import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_sum_where(df, value_column, cond_column, cond_threshold, bounds, max_length, epsilon,
                 n_is_public=True):
    """Sum of one column over the rows where another numeric column exceeds a public threshold,
    using an expression-level filter inside the DP aggregate. The condition is a public rule;
    removing rows never raises the sensitivity. Default: replace-one (n public), unit_of(changes=1),
    charged U - L. n_is_public=False: add/remove-one, unit_of(contributions=1), charged max(|L|, |U|).

    Returns (private_conditional_sum, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1) if n_is_public else dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = context.query().select(
        pl.col(value_column).cast(float)
        .filter(pl.col(cond_column) > cond_threshold)   # row-by-row predicate, then aggregate
        .dp.sum(bounds))
    print(query.summarize().to_dicts())
    value = query.release().collect().item()
    return value, context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(200)], "level": [i % 7 for i in range(200)]})
    _, eps_r1 = dp_sum_where(df, "value", "level", 3, (-5.0, 10.0), 1000, 1.0)
    _, eps_ar = dp_sum_where(df, "value", "level", 3, (-5.0, 10.0), 1000, 1.0, n_is_public=False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
