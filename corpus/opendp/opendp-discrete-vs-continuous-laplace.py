import opendp.prelude as dp

dp.enable_features("contrib")


def dp_int_count_and_float_sum(df, column, bounds, epsilon):
    """Shows how the space picks the Laplace variant: an int count gets discrete Laplace noise,
    a float sum gets OpenDP's float-safe continuous Laplace. Replace-one (n public): sized domain,
    d_in = 2 (add/remove-one, n private: unsized domain, d_in = 1).

    Returns ((int_count, float_sum), epsilon_spent).
    """
    values = df[column].astype(float).tolist()
    d_in = 2                                                 # one changed row = symmetric distance 2
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()
    count = space >> dp.t.then_count()                       # output space: int, absolute distance
    total = space >> dp.t.then_clamp(bounds) >> dp.t.then_sum()  # output space: float, absolute distance

    def int_noise(scale):   # explicit stand-alone form; `count >> then_laplace(scale)` is the same
        return count >> dp.m.make_laplace(dp.atom_domain(T=int), dp.absolute_distance(T=int), scale)

    def float_noise(scale):
        return total >> dp.m.make_laplace(
            dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float), scale)

    half = epsilon / 2
    count_meas = dp.binary_search_chain(int_noise, d_in=d_in, d_out=half)
    sum_meas = dp.binary_search_chain(float_noise, d_in=d_in, d_out=half)
    comp = dp.c.make_composition([count_meas, sum_meas])
    epsilon_spent = comp.map(d_in)
    dp_count, dp_sum = comp(values)
    return (dp_count, dp_sum), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    (dp_count, dp_sum), epsilon_spent = dp_int_count_and_float_sum(df, "value", (-5.0, 5.0), 1.0)
    assert isinstance(dp_count, int) and isinstance(dp_sum, float)
    return {"epsilon": float(epsilon_spent)}
