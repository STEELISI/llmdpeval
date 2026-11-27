import opendp.prelude as dp

dp.enable_features("contrib")


def dp_count_with_checks(df, column, value, epsilon, alpha=0.05):
    """DP count of rows where df[column] == value that checks its own sensitivity, privacy loss and
    accuracy. Replace-one neighbours (n public): sized domain, d_in = 2 (one changed row).

    Returns ({"count", "accuracy"}, epsilon_spent); the count is within +-accuracy w.p. 1 - alpha.
    """
    flags = (df[column].astype(str) == str(value)).astype(int).tolist()   # row-wise 0/1, no budget
    space = dp.vector_domain(dp.atom_domain(T=int), size=len(flags)), dp.symmetric_distance()
    count = space >> dp.t.then_clamp((0, 1)) >> dp.t.then_sum()
    print("sensitivity at d_in=2:", count.map(2))      # what OpenDP charges (1), not an assumed value

    scale = dp.binary_search_param(lambda s: count >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    meas = count >> dp.m.then_laplace(scale)
    epsilon_spent = meas.map(2)                        # the guarantee of the measurement invoked
    assert meas.check(2, epsilon) and epsilon_spent <= epsilon
    # Integer count -> discrete Laplace -> use the discrete accuracy helper.
    accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=scale, alpha=alpha)
    private_count = meas(flags)
    return {"count": private_count, "accuracy": accuracy}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "c"] * 30})
    _, epsilon_spent = dp_count_with_checks(df, "group", "a", epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
