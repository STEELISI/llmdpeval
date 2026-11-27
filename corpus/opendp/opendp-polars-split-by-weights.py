import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_weighted_budget_queries(df, column, bounds, weights, alpha, max_length, epsilon, n_is_public=True):
    """Polars Context whose total epsilon is split unevenly across queries (split_by_weights):
    the query that needs more accuracy gets the bigger share. summarize(alpha) shows the
    (1 - alpha) error bound of each statistic before anything is released.
    One person = one row. Default: replace-one (n public),
    unit_of(changes=1). n_is_public=False: add/remove-one, unit_of(contributions=1).
    Under changes=1 the length is public: the count is exact and the mean is a sized mean.

    weights: one positive weight per query, in order (here 2: a count and a mean).
    Returns ((count, mean), epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1) if n_is_public else dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_by_weights=weights,                 # instead of split_evenly_over
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    q_count = context.query().select(dp.len(signed=True))
    print(q_count.summarize(alpha=alpha).to_dicts())   # "accuracy" column = error bound at 1 - alpha
    count = q_count.release().collect().item()

    q_mean = context.query().select(pl.col(column).cast(float).dp.mean(bounds))
    print(q_mean.summarize().to_dicts())               # mean = noisy sum / noisy length
    mean = q_mean.release().collect().item()
    print(context.remaining_privacy_loss())            # 0 after the last weighted query
    return (count, mean), context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(200)]})
    _, eps_r1 = dp_weighted_budget_queries(df, "value", (-5.0, 5.0), [1, 3], 0.05, 1000, 1.0)
    _, eps_ar = dp_weighted_budget_queries(df, "value", (-5.0, 5.0), [1, 3], 0.05, 1000, 1.0, False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
