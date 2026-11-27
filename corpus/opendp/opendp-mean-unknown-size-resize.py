import opendp.prelude as dp

dp.enable_features("contrib")


def dp_mean_private_size(df, column, bounds, impute_value, epsilon, count_share=0.5):
    """DP mean when n is private: the add/remove-one alternative (d_in = 1). When n is public
    (replace-one), use a sized domain with d_in = 2 instead (opendp-mean-sized-bounded). Release a DP count, resize to it,
    then release a sized mean. count_share is the public part of epsilon spent on the count.

    Returns (private_mean, epsilon_spent).
    """
    space = dp.vector_domain(dp.atom_domain(T=float, nan=True)), dp.symmetric_distance()
    values = df[column].astype(float).tolist()
    count_meas = dp.binary_search_chain(
        lambda s: space >> dp.t.then_count() >> dp.m.then_laplace(s), d_in=1, d_out=epsilon * count_share)
    dp_n = max(count_meas(values), 1)                   # the DP count, never len(values)

    mean_pre = (space >> dp.t.then_impute_constant(impute_value) >> dp.t.then_clamp(bounds)
                >> dp.t.then_resize(size=dp_n, constant=impute_value) >> dp.t.then_mean())
    mean_meas = dp.binary_search_chain(
        lambda s: mean_pre >> dp.m.then_laplace(s), d_in=1, d_out=epsilon * (1 - count_share))
    private_mean = mean_meas(values)
    # Total = count + mean by sequential composition; let OpenDP add them.
    epsilon_spent = dp.c.make_composition([count_meas, mean_meas]).map(1)
    return private_mean, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 10) for i in range(100)] + [float("nan")]})
    _, epsilon_spent = dp_mean_private_size(df, "value", (0.0, 10.0), impute_value=5.0, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
