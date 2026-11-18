import opendp.prelude as dp

dp.enable_features("contrib")


def dp_count_and_sum_zcdp(df, column, bounds, epsilon, delta, weights=(1.0, 1.0)):
    """Several Gaussian releases (here a count and a bounded sum) composed in zCDP, then converted
    to (epsilon, delta) once at the end. Tighter than converting each query and adding epsilons.
    Replace-one (n public): sized domain, d_in = 2 (add/remove-one, n private: unsized, d_in = 1).

    weights: relative noise scale per query (public). Returns ((count, sum), (epsilon_spent, delta_spent)).
    """
    values = df[column].astype(float).tolist()
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()
    pre_count = space >> dp.t.then_count()
    pre_sum = space >> dp.t.then_clamp(bounds) >> dp.t.then_sum()
    span = bounds[1] - bounds[0]                 # the sum's noise must grow with the value range

    def make(scale):
        comp = dp.c.make_composition([pre_count >> dp.m.then_gaussian(scale * weights[0]),
                                      pre_sum >> dp.m.then_gaussian(scale * weights[1] * span)])
        # composition is in rho (zCDP); convert the total once, then fix delta
        return dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(comp), delta=delta)

    meas = dp.binary_search_chain(make, d_in=2, d_out=(epsilon, delta))
    dp_count, dp_sum = meas(values)
    return (dp_count, dp_sum), meas.map(2)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, (eps, _) = dp_count_and_sum_zcdp(df, "value", (-5.0, 5.0), epsilon=1.0, delta=1e-6)
    return {"epsilon": float(eps)}
