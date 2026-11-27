import opendp.prelude as dp

dp.enable_features("contrib")


def dp_sum_clamped(df, column, bounds, impute_value, epsilon):
    """DP sum of a float column that may contain NaN. bounds and impute_value must be public
    (domain knowledge), never min()/max() of the data. Returns (private_sum, epsilon_spent).
    Replace-one neighbours (n public): sized domain, d_in = 2; the sum is charged U - L.
    """
    lower, upper = bounds
    assert lower <= impute_value <= upper              # the imputed constant must lie in the bounds
    values = df[column].astype(float).tolist()
    space = dp.vector_domain(dp.atom_domain(T=float, nan=True), size=len(values)), dp.symmetric_distance()
    # Impute and clamp inside OpenDP so the stability maps know the bounds.
    pre = space >> dp.t.then_impute_constant(impute_value) >> dp.t.then_clamp(bounds) >> dp.t.then_sum()
    print("sum sensitivity implied by the bounds:", pre.map(2))   # replace-one = distance 2

    meas = dp.binary_search_chain(lambda s: pre >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)
    private_sum = meas(values)
    return private_sum, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [1.0, -7.0, float("nan"), 12.0, 3.0] * 10})
    _, epsilon_spent = dp_sum_clamped(df, "value", (-5.0, 10.0), impute_value=0.0, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
