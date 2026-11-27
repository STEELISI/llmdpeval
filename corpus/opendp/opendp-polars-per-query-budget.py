import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_queries_with_own_budget(df, column, bounds, query_epsilons, max_length, epsilon, n_is_public=True):
    """Polars Context without a fixed split: each query names its own epsilon in
    context.query(epsilon=...), and the context refuses a query that would overspend the total.
    One person = one row. Default: replace-one (n public),
    unit_of(changes=1). n_is_public=False: add/remove-one, unit_of(contributions=1).

    query_epsilons: [eps for the count, eps for the sum]; their total must not exceed epsilon.
    Returns ((count, sum), epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1) if n_is_public else dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=epsilon),     # no split_evenly_over / split_by_weights
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    count = context.query(epsilon=query_epsilons[0]).select(dp.len(signed=True)).release().collect().item()
    print(context.remaining_privacy_loss())           # budget left for later queries
    total = (context.query(epsilon=query_epsilons[1])
             .select(pl.col(column).cast(float).dp.sum(bounds)).release().collect().item())
    return (count, total), context.current_privacy_loss()


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(200)]})
    _, eps_r1 = dp_queries_with_own_budget(df, "value", (-5.0, 5.0), [0.3, 0.6], 1000, 1.0)
    _, eps_ar = dp_queries_with_own_budget(df, "value", (-5.0, 5.0), [0.3, 0.6], 1000, 1.0, False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
