import opendp.prelude as dp

dp.enable_features("contrib")


def dp_mean_public_size(df, column, bounds, size, epsilon):
    """DP mean when n (size) is public: bounded DP, neighbours replace one record, so d_in = 2.

    bounds are public; values are clamped inside OpenDP. Returns (private_mean, epsilon_spent).
    """
    space = (dp.vector_domain(dp.atom_domain(T=float, nan=False), size=size),
             dp.symmetric_distance())
    mean = space >> dp.t.then_clamp(bounds) >> dp.t.then_mean()   # divides by size, not len(data)
    print("mean sensitivity at d_in=2:", mean.map(2))

    meas = dp.binary_search_chain(lambda s: mean >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)
    assert epsilon_spent <= epsilon
    values = df[column].astype(float).tolist()
    assert len(values) == size, "data length must equal the public size"
    return meas(values), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 10) for i in range(100)]})
    _, epsilon_spent = dp_mean_public_size(df, "value", (0.0, 10.0), size=100, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
