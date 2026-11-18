import opendp.prelude as dp

dp.enable_features("contrib")


def dp_sum_and_mean(df, column, bounds, impute_value, epsilon, n_is_public=True):
    """Two releases (a size statistic, then a mean) of one float column under one total epsilon
    that the Context enforces.

    Default: replace-one (n public): sized domain, privacy unit = symmetric distance 2, returns
    (sum, mean). n_is_public=False: add/remove-one (n private): unit 1, returns (DP count, mean)
    with the mean resized to the DP count. Returns ((first, mean), epsilon_spent).
    """
    values = df[column].astype(float).fillna(impute_value).tolist()
    atom = dp.atom_domain(T=float, nan=False)
    domain = dp.vector_domain(atom, size=len(values)) if n_is_public else dp.vector_domain(atom)
    context = dp.Context.compositor(
        data=values,
        privacy_unit=dp.unit_of(contributions=2 if n_is_public else 1),   # 2 = one changed row
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=2,                           # exactly two releases
        domain=domain,
    )
    if n_is_public:
        first = context.query().clamp(bounds).sum().laplace().release()       # charged U - L
        mean_query = context.query().clamp(bounds).mean().laplace()           # (U - L) / n
    else:
        first = context.query().count().laplace().release()
        mean_query = (context.query().clamp(bounds)
                      .resize(size=max(first, 1), constant=impute_value)      # the DP count
                      .mean().laplace())
    print("scale chosen for the mean:", mean_query.param())
    dp_mean = mean_query.release()
    return (first, dp_mean), context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(200)]})
    _, eps_r1 = dp_sum_and_mean(df, "value", (-5.0, 5.0), impute_value=0.0, epsilon=1.0)
    _, eps_ar = dp_sum_and_mean(df, "value", (-5.0, 5.0), 0.0, epsilon=1.0, n_is_public=False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
