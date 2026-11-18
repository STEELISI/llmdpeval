import opendp.prelude as dp

dp.enable_features("contrib")


def dp_quantiles(df, column, candidates, alphas, epsilon):
    """DP quantiles (e.g. alphas=[0.25, 0.5, 0.75]) by the exponential mechanism over a public
    candidate grid. Replace-one neighbours (bounded DP, n public); all quantiles share one epsilon.

    Returns (list_of_quantiles, epsilon_spent).
    """
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()

    def make(scale):
        return dp.c.make_composition([
            dp.m.make_private_quantile(*space, dp.max_divergence(), candidates=candidates,
                                       alpha=a, scale=scale)
            for a in alphas
        ])

    # Replace-one = one removal plus one addition: symmetric distance 2. One shared scale such that
    # the whole composition maps to at most epsilon at d_in = 2.
    meas = dp.binary_search_chain(make, d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)
    assert epsilon_spent <= epsilon + 1e-9
    private_quantiles = meas(df[column].astype(float).tolist())
    return sorted(private_quantiles), epsilon_spent   # sorting is post-processing


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 50) for i in range(200)]})
    grid = [float(c) for c in range(0, 51)]
    qs, epsilon_spent = dp_quantiles(df, "value", grid, [0.25, 0.5, 0.75], epsilon=1.0)
    assert len(qs) == 3
    return {"epsilon": float(epsilon_spent)}
