import opendp.prelude as dp

dp.enable_features("contrib")


def dp_sum_searched_scale(df, column, bounds, epsilon, d_in=2):
    """DP sum of a clamped float column; the noise scale is searched from epsilon by OpenDP.

    Replace-one neighbours (n public): d_in = 2, one changed row = symmetric distance 2 (2*K if a
    person has up to K rows). Sized domain, so the sum is charged U - L. Returns (private_sum, epsilon_spent).
    """
    values = df[column].astype(float).tolist()
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()

    def make(scale):                                   # the pipeline as a function of one number
        return space >> dp.t.then_clamp(bounds) >> dp.t.then_sum() >> dp.m.then_laplace(scale)

    # Smallest scale with make(scale).map(d_in) <= epsilon; no hand formula.
    scale = dp.binary_search_param(make, d_in=d_in, d_out=epsilon)
    meas = make(scale)
    epsilon_spent = meas.map(d_in)
    assert epsilon_spent <= epsilon
    private_sum = meas(values)
    return private_sum, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(50)]})
    _, epsilon_spent = dp_sum_searched_scale(df, "value", (-5.0, 5.0), epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
