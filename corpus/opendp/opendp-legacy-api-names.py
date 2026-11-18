import opendp.prelude as dp     # make_base_laplace / make_bounded_sum / make_sized_bounded_mean are gone

dp.enable_features("contrib")    # not "floating-point" (deprecated alias of "idealized-numerics")


def dp_bounded_mean_modern(df, column, bounds, size, impute_value, epsilon):
    """Modern 0.16 form of the old make_sized_bounded_mean(size, bounds) >> make_base_laplace(scale).

    Bounds and size live on the domain. Replace-one (n public): size is the public row count, the
    domain is sized, d_in = 2. (Add/remove-one, n private: unsized domain, then_resize to a DP
    count before then_mean, d_in = 1.) Returns (private_mean, epsilon_spent).
    """
    values = df[column].astype(float).fillna(impute_value).tolist()
    assert len(values) == size, "n is public under replace-one and must match the data"
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=size), dp.symmetric_distance()
    pre = space >> dp.t.then_clamp(bounds) >> dp.t.then_mean()   # mean charged (U - L) / size
    # make_laplace / then_laplace replaces make_base_laplace and make_base_discrete_laplace.
    # Re-search the scale: an old hand-computed scale does not carry over.
    meas = dp.binary_search_chain(lambda s: pre >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)                       # replace-one = symmetric distance 2
    private_mean = meas(values)
    return private_mean, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, epsilon_spent = dp_bounded_mean_modern(df, "value", (-5.0, 5.0), size=100, impute_value=0.0,
                                              epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
