import opendp.prelude as dp

dp.enable_features("contrib")


def dp_adaptive_count_then_sum(df, column, bounds, epsilon):
    """Two queries in one adaptive session: a DP count, then a DP clamped sum.

    Replace-one neighbours (n public): one changed row is symmetric distance 2, so d_in = 2 everywhere.
    The compositor enforces each query's slot and the total. Returns ((count, sum), epsilon_spent).
    """
    values = df[column].astype(float).tolist()
    d_in = 2                                         # replace-one = symmetric distance 2
    # Sized domain (n public): the sum is charged U - L at d_in = 2.
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()
    d_mids = [epsilon / 2, epsilon / 2]              # budget plan, fixed before seeing data
    comp = dp.c.make_adaptive_composition(*space, dp.max_divergence(), d_in=d_in, d_mids=d_mids)
    qbl = comp(values)                               # a queryable session, not an answer

    # Each query searches its scale against its own slot, with the same d_in as the compositor.
    # (OpenDP charges the count 2 at d_in = 2; for a filtered count use a 0/1 sum, charged 1.)
    count_meas = dp.binary_search_chain(
        lambda s: space >> dp.t.then_count() >> dp.m.then_laplace(s), d_in=d_in, d_out=d_mids[0])
    dp_count = qbl(count_meas)
    # A later query may depend on earlier answers (e.g. skip the sum if dp_count is tiny).
    sum_meas = dp.binary_search_chain(
        lambda s: space >> dp.t.then_clamp(bounds) >> dp.t.then_sum() >> dp.m.then_laplace(s),
        d_in=d_in, d_out=d_mids[1])
    dp_sum = qbl(sum_meas)
    epsilon_spent = comp.map(d_in)
    return (dp_count, dp_sum), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, epsilon_spent = dp_adaptive_count_then_sum(df, "value", (-5.0, 5.0), epsilon=1.0)
    assert epsilon_spent <= 1.0 + 1e-9
    return {"epsilon": float(epsilon_spent)}
