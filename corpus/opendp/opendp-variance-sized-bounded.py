import opendp.prelude as dp

dp.enable_features("contrib")


def dp_variance_public_size(df, column, bounds, size, epsilon, ddof=1):
    """DP sample variance when n (size) and bounds are public: bounded DP, d_in = 2.

    Returns (private_variance, epsilon_spent). A DP std is sqrt(max(variance, 0)) afterwards.
    """
    space = (dp.vector_domain(dp.atom_domain(T=float, nan=False), size=size),
             dp.symmetric_distance())
    var = space >> dp.t.then_clamp(bounds) >> dp.t.then_variance(ddof=ddof)
    print("variance sensitivity at d_in=2:", var.map(2))   # grows with (U - L)^2 / N

    meas = dp.binary_search_chain(lambda s: var >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)
    assert epsilon_spent <= epsilon
    values = df[column].astype(float).tolist()
    assert len(values) == size, "data length must equal the public size"
    return meas(values), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 10) for i in range(100)]})
    _, epsilon_spent = dp_variance_public_size(df, "value", (0.0, 10.0), size=100, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
