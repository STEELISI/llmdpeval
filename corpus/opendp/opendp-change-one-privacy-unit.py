import opendp.prelude as dp

dp.enable_features("contrib")


def dp_mean_replace_one(df, column, bounds, size, epsilon):
    """Bounded mean under "replace-one" (bounded DP): neighbours have the same public size n and
    differ in one changed record. Express the unit as change_one_distance with d_in = 1, then convert
    to the add/remove metric the aggregators use (one change = 2 add/remove).

    Returns (private_mean, epsilon_spent).
    """
    atom = dp.atom_domain(bounds=bounds, nan=False)          # bounds are public, data pre-clamped
    space = dp.vector_domain(atom, size=size), dp.change_one_distance()
    pre = space >> dp.t.then_metric_unbounded() >> dp.t.then_mean()
    meas = dp.binary_search_chain(lambda s: pre >> dp.m.then_laplace(s), d_in=1, d_out=epsilon)
    values = df[column].astype(float).clip(*bounds).tolist()  # row-wise clamp with PUBLIC bounds
    assert len(values) == size, "size must be the public n of the dataset"
    return meas(values), meas.map(1)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 10) for i in range(100)]})
    _, epsilon_spent = dp_mean_replace_one(df, "value", (0.0, 10.0), 100, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
