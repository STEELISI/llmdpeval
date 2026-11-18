import opendp.prelude as dp

dp.enable_features("contrib")


def dp_sum_with_missing(df, column, bounds, how, epsilon):
    """Bounded sum of a float column that has missing values (NaN). Aggregators need non-null data,
    so impute first: "constant" fills with a public value, "uniform" draws from the public bounds,
    "drop" removes the rows. Replace-one (n public): sized domain, d_in = 2; the sum is charged U - L
    after an imputer, more after "drop" (the length becomes data-dependent). Add/remove-one (n private):
    unsized domain, d_in = 1.

    Returns (private_sum, epsilon_spent).
    """
    values = df[column].astype(float).tolist()
    space = dp.vector_domain(dp.atom_domain(T=float, nan=True), size=len(values)), dp.symmetric_distance()
    imputer = {"constant": dp.t.then_impute_constant((bounds[0] + bounds[1]) / 2),
               "uniform": dp.t.then_impute_uniform_float(bounds),
               "drop": dp.t.then_drop_null()}[how]
    pre = space >> imputer >> dp.t.then_clamp(bounds) >> dp.t.then_sum()
    meas = dp.binary_search_chain(lambda s: pre >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    return meas(values), meas.map(2)


def _test():
    import pandas as pd
    vals = [float(i % 10) for i in range(90)] + [float("nan")] * 10
    df = pd.DataFrame({"value": vals})
    for how in ("constant", "uniform", "drop"):
        _, epsilon_spent = dp_sum_with_missing(df, "value", (-5.0, 10.0), how, epsilon=1.0)
        assert epsilon_spent <= 1.0
    return {"epsilon": float(epsilon_spent)}
