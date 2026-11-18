import opendp.prelude as dp

dp.enable_features("contrib")


def dp_count_by_unknown_keys(df, column, epsilon, delta):
    """Noisy count per key when no public key list exists. Keys with small noisy counts are
    suppressed, so the guarantee is (epsilon, delta). Replace-one neighbours (bounded DP,
    n public): symmetric distance d_in = 2.

    Returns ({key: count}, (epsilon_spent, delta_spent)).
    """
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    counts = space >> dp.t.then_count_by()

    def make(scale, threshold):
        return counts >> dp.m.then_laplace_threshold(scale=scale, threshold=threshold)

    # Epsilon depends on the scale, delta on the threshold: search them one at a time.
    # (binary_search_chain on an (eps, delta) tuple fails with "unknown ordering".)
    scale = dp.binary_search(lambda s: make(s, 10**9).map(2)[0] <= epsilon, bounds=(1e-6, 1e6))
    threshold = dp.binary_search(lambda t: make(scale, t).map(2)[1] <= delta, bounds=(2, 10**9))   # threshold >= l-inf sensitivity (2 under replace-one)
    meas = make(scale, threshold)
    epsilon_spent, delta_spent = meas.map(2)   # replace-one
    assert epsilon_spent <= epsilon and delta_spent <= delta
    released = meas(df[column].astype(str).tolist())   # only keys whose noisy count passed
    return released, (epsilon_spent, delta_spent)


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 60 + ["b"] * 30 + ["c"]})
    _, (epsilon_spent, _) = dp_count_by_unknown_keys(df, "group", epsilon=1.0, delta=1e-6)
    return {"epsilon": float(epsilon_spent)}
