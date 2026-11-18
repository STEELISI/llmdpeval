import opendp.prelude as dp

dp.enable_features("contrib")


def dp_queries_until_budget(df, column, bounds, size, queries, epsilon_per_query, epsilon):
    """Privacy filter: a fully adaptive odometer capped at a total epsilon. Queries are answered
    until the next one would push the total past the cap; then the filter refuses.
    Replace-one (n public): sized domain, d_in = 2 (add/remove-one, n private: unsized, d_in = 1).

    queries: list of names from {"count", "sum", "mean"}; bounds and size are public.
    Returns ({query: value}, epsilon_spent).
    """
    values = df[column].astype(float).tolist()
    assert len(values) == size, "n is public under replace-one and must match the data"
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=size), dp.symmetric_distance()
    odometer = dp.c.make_fully_adaptive_composition(*space, output_measure=dp.max_divergence())
    meas_filter = dp.c.make_privacy_filter(odometer, d_in=2, d_out=epsilon)
    qbl = meas_filter(values)

    clamp = space >> dp.t.then_clamp(bounds)
    pre = {"count": space >> dp.t.then_count(),
           "sum": clamp >> dp.t.then_sum(),
           "mean": clamp >> dp.t.then_mean()}            # sized: charged (U - L) / size
    released, epsilon_spent = {}, 0.0
    for name in queries:
        meas = dp.binary_search_chain(lambda s: pre[name] >> dp.m.then_laplace(s),
                                      d_in=2, d_out=epsilon_per_query)
        try:
            released[name] = qbl(meas)
        except dp.OpenDPException:      # "filter is now exhausted": stop asking
            break
        epsilon_spent = qbl.privacy_loss(2)   # once exhausted, the queryable refuses this call too
    return released, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    out, epsilon_spent = dp_queries_until_budget(df, "value", (-5.0, 5.0), 100, ["count", "sum", "mean"],
                                                 epsilon_per_query=0.4, epsilon=1.0)
    assert len(out) == 2   # the third query would exceed the cap
    return {"epsilon": float(epsilon_spent)}
