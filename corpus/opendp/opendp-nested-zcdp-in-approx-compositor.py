import opendp.prelude as dp

dp.enable_features("contrib")


def dp_nested_gaussian_and_laplace(df, column, bounds, epsilon, delta):
    """An (epsilon, delta) adaptive compositor with two slots: slot 1 holds a nested zCDP compositor
    (Gaussian sum + Gaussian count), slot 2 a pure-DP Laplace count. Replace-one (n public): sized
    domain, d_in = 2 throughout (add/remove-one, n private: unsized, d_in = 1).

    bounds are public. Returns ((gauss_sum, gauss_count, laplace_count), (epsilon_spent, delta_spent)).
    """
    values = df[column].astype(float).tolist()
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()
    root = dp.c.make_adaptive_composition(*space, output_measure=dp.approximate(dp.max_divergence()),
                                          d_in=2, d_mids=[(epsilon / 2, delta), (epsilon / 2, 0.0)])
    qbl_root = root(values)

    # Child: zCDP compositor over two queries (rho split evenly), converted to (eps, delta).
    def make_child(rho_each):
        child = dp.c.make_adaptive_composition(*space, output_measure=dp.zero_concentrated_divergence(),
                                               d_in=2, d_mids=[rho_each, rho_each])
        return dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(child), delta=delta)
    rho_each = dp.binary_search_param(make_child, d_in=2, d_out=(epsilon / 2, delta), T=float)
    qbl_child = qbl_root(make_child(rho_each))

    m_sum = dp.binary_search_chain(lambda s: space >> dp.t.then_clamp(bounds) >> dp.t.then_sum()
                                   >> dp.m.then_gaussian(s), d_in=2, d_out=rho_each)
    m_cnt = dp.binary_search_chain(lambda s: space >> dp.t.then_count() >> dp.m.then_gaussian(s),
                                   d_in=2, d_out=rho_each)
    gauss_sum, gauss_count = qbl_child(m_sum), qbl_child(m_cnt)

    # Slot 2: a pure-DP measurement must be cast to approx-DP (delta = 0) to enter the root.
    m_lap = dp.binary_search_chain(lambda s: space >> dp.t.then_count() >> dp.m.then_laplace(s),
                                   d_in=2, d_out=epsilon / 2)
    laplace_count = qbl_root(dp.c.make_approximate(m_lap))
    return (gauss_sum, gauss_count, laplace_count), root.map(2)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, (eps, _) = dp_nested_gaussian_and_laplace(df, "value", (-5.0, 5.0), epsilon=1.0, delta=1e-6)
    return {"epsilon": float(eps)}
