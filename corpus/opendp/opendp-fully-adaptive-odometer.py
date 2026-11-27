import opendp.prelude as dp

dp.enable_features("contrib")


def dp_odometer_count_then_sum(df, column, bounds, epsilon_count, epsilon_sum):
    """Fully adaptive composition (an odometer): queries and their privacy losses are chosen
    one at a time, and the odometer reports the running total. Replace-one (n public): sized
    domain, d_in = 2 (add/remove-one, n private: unsized domain, d_in = 1).

    bounds: public (L, U) for the column. Returns ((count, sum), epsilon_spent).
    """
    values = df[column].astype(float).tolist()
    d_in = 2                                  # one changed row = symmetric distance 2
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()
    odometer = dp.c.make_fully_adaptive_composition(*space, output_measure=dp.max_divergence())
    qbl = odometer(values)

    # Each query gets its scale from its own epsilon; it can depend on earlier answers.
    m_count = dp.binary_search_chain(
        lambda s: space >> dp.t.then_count() >> dp.m.then_laplace(s), d_in=d_in, d_out=epsilon_count)
    dp_count = qbl(m_count)
    m_sum = dp.binary_search_chain(
        lambda s: space >> dp.t.then_clamp(bounds) >> dp.t.then_sum() >> dp.m.then_laplace(s),
        d_in=d_in, d_out=epsilon_sum)
    dp_sum = qbl(m_sum)

    epsilon_spent = qbl.privacy_loss(d_in)   # running total over every query answered so far
    return (dp_count, dp_sum), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, epsilon_spent = dp_odometer_count_then_sum(df, "value", (-5.0, 5.0), 0.5, 0.5)
    return {"epsilon": float(epsilon_spent)}
