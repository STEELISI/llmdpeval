import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_count_sum_gaussian_context(df, column, bounds, max_length, rho, delta, n_is_public=True):
    """Polars Context under zCDP: loss_of(rho=...) makes every query use Gaussian noise instead of
    Laplace. The context's total rho is converted to (epsilon, delta) at the end.
    One person = one row. Default: replace-one (n public),
    unit_of(changes=1). n_is_public=False: add/remove-one, unit_of(contributions=1).

    bounds, max_length (public upper bound on rows) and delta are public.
    Returns (one-row result dict, (epsilon_spent, delta_spent)).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1) if n_is_public else dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(rho=rho),          # rho -> Gaussian; epsilon -> Laplace
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    query = context.query().select(
        dp.len(signed=True),                       # signed=True keeps negative noisy counts
        pl.col(column).cast(float).dp.sum(bounds),
    )
    print(query.summarize(alpha=0.05).to_dicts())  # distribution "Integer/Float Gaussian"
    row = query.release().collect().to_dicts()[0]
    # The accountant measures rho; convert the total to (epsilon, delta) with the combinators.
    epsilon_spent, delta_spent = dp.c.make_fix_delta(
        dp.c.make_zCDP_to_approxDP(context.accountant), delta=delta).map(context.d_in)
    return row, (epsilon_spent, delta_spent)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(200)]})
    _, (e1, _) = dp_count_sum_gaussian_context(df, "value", (-5.0, 5.0), 1000, rho=0.02, delta=1e-6)
    _, (e2, _) = dp_count_sum_gaussian_context(df, "value", (-5.0, 5.0), 1000, 0.02, 1e-6, False)
    return {"epsilon": float(max(e1, e2))}
