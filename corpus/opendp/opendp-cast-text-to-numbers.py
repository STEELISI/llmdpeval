import opendp.prelude as dp

dp.enable_features("contrib")


def dp_mean_from_text_column(df, column, bounds, size, impute_value, epsilon):
    """DP mean of numbers stored as text (e.g. read from a raw file). Cast first; failed casts become
    null, so impute before clamping. Replace-one neighbours (n public): sized domain, d_in = 2.

    size: the public row count (must equal the number of rows). Returns (private_mean, epsilon_spent).
    """
    values = df[column].astype(str).tolist()
    assert len(values) == size, "n is public under replace-one and must match the data"
    space = dp.vector_domain(dp.atom_domain(T=str), size=size), dp.symmetric_distance()
    pre = (space
           >> dp.t.then_cast(float)                    # Option[float]: failed parse -> None
           >> dp.t.then_impute_constant(impute_value)  # public fill value, inside the bounds
           >> dp.t.then_clamp(bounds)                  # sized + bounded: mean charged (U - L) / n
           >> dp.t.then_mean())
    meas = dp.binary_search_chain(lambda s: pre >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    return meas(values), meas.map(2)                  # replace-one = symmetric distance 2


def _test():
    import pandas as pd
    df = pd.DataFrame({"raw": [str(i % 10) for i in range(95)] + ["", "n/a", "?", "1e2", "x"]})
    _, epsilon_spent = dp_mean_from_text_column(df, "raw", (-5.0, 10.0), 100, 5.0, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
