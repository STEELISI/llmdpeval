import opendp.prelude as dp

dp.enable_features("contrib")


def dp_count_where_public_size(df, column, value, size, epsilon):
    """Bounded-DP count of rows with df[column] == value, when n (size) is public.

    Replace-one neighbours, so d_in = 2. A sized 0/1 sum has sensitivity 1 at d_in = 2;
    then_count on a sized domain would be charged 2. Returns (private_count, epsilon_spent).
    """
    space = (dp.vector_domain(dp.atom_domain(bounds=(0, 1)), size=size),
             dp.symmetric_distance())
    total = space >> dp.t.then_sum()
    print("sized 0/1 sum sensitivity at d_in=2:", total.map(2))

    meas = dp.binary_search_chain(lambda s: total >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)                      # never d_in=1 on a sized domain
    indicator = (df[column] == value).astype(int).tolist()
    assert len(indicator) == size, "data length must equal the public size"
    return meas(indicator), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c"] * 25})
    _, epsilon_spent = dp_count_where_public_size(df, "group", "a", size=100, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
