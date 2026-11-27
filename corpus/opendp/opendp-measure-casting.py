import opendp.prelude as dp

dp.enable_features("contrib")


def dp_laplace_count_gaussian_sum(df, column, bounds, epsilon, delta):
    """A Laplace count and a Gaussian sum in one composition with one (epsilon, delta) guarantee.

    Cast the Laplace piece to zCDP, compose in zCDP, convert the whole composition once.
    Replace-one (n public): sized domain, d_in = 2 (add/remove-one, n private: unsized, d_in = 1).
    Returns ((count, sum), (epsilon_spent, delta_spent)).
    """
    values = df[column].astype(float).tolist()
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()
    count = space >> dp.t.then_count()
    total = space >> dp.t.then_clamp(bounds) >> dp.t.then_sum()
    sum_sens = total.map(2)                    # read from OpenDP; noise scales with each query

    def make(s):
        lap = dp.c.make_pureDP_to_zCDP(count >> dp.m.then_laplace(s))
        gauss = total >> dp.m.then_gaussian(s * sum_sens)
        comp = dp.c.make_composition([lap, gauss])           # both now measure rho
        return dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(comp), delta=delta)

    meas = dp.binary_search_chain(make, d_in=2, d_out=(epsilon, delta))
    epsilon_spent, delta_spent = meas.map(2)
    assert epsilon_spent <= epsilon and delta_spent <= delta
    dp_count, dp_sum = meas(values)
    return (dp_count, dp_sum), (epsilon_spent, delta_spent)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, (epsilon_spent, _) = dp_laplace_count_gaussian_sum(df, "value", (-5.0, 5.0), 1.0, 1e-6)
    return {"epsilon": float(epsilon_spent)}
