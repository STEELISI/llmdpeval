import opendp.prelude as dp            # the prelude, not `import opendp as dp`

# Most constructors are "contrib" and raise until you opt in. Add other flags only when an
# error message names them (e.g. "honest-but-curious" for user-defined functions).
dp.enable_features("contrib")


def dp_count_rows(df, column, value, epsilon):
    """DP count of rows where df[column] == value, using the explicit namespaces: dp.t
    (transformations), dp.m (measurements). Replace-one (n public): sized domain, d_in = 2.

    Returns (private_count, epsilon_spent).
    """
    flags = (df[column].astype(str) == str(value)).astype(int).tolist()
    input_domain = dp.vector_domain(dp.atom_domain(T=int), size=len(flags))
    input_metric = dp.symmetric_distance()
    count = dp.t.make_clamp(input_domain, input_metric, bounds=(0, 1)) >> dp.t.then_sum()
    meas = dp.binary_search_chain(lambda s: count >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)                        # replace-one = symmetric distance 2
    private_count = meas(flags)
    return private_count, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b"] * 25})
    _, epsilon_spent = dp_count_rows(df, "group", "a", epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
