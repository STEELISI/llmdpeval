import opendp.prelude as dp

dp.enable_features("contrib")


def dp_gaussian_sum(df, column, bounds, size, epsilon, delta):
    """(epsilon, delta)-DP sum and mean with Gaussian noise. size (n) and bounds are public, so the
    domain is sized and neighbours replace one record (d_in = 2). The mean is sum / size (post-processing).

    Returns ((private_sum, private_mean), (epsilon_spent, delta_spent)).
    """
    space = (dp.vector_domain(dp.atom_domain(T=float, nan=False), size=size),
             dp.symmetric_distance())
    pre = space >> dp.t.then_clamp(bounds) >> dp.t.then_sum()

    def make(scale):
        # Gaussian measures rho (zCDP); convert to a privacy profile, then fix delta.
        return dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(pre >> dp.m.then_gaussian(scale)), delta=delta)

    meas = dp.binary_search_chain(make, d_in=2, d_out=(epsilon, delta))
    epsilon_spent, delta_spent = meas.map(2)
    assert epsilon_spent <= epsilon and delta_spent <= delta
    private_sum = meas(df[column].astype(float).tolist())   # list length must equal size
    return (private_sum, private_sum / size), (epsilon_spent, delta_spent)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, (epsilon_spent, _) = dp_gaussian_sum(df, "value", (-5.0, 5.0), size=100, epsilon=1.0, delta=1e-6)
    return {"epsilon": float(epsilon_spent)}
