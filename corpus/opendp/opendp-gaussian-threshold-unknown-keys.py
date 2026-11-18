import opendp.prelude as dp

dp.enable_features("contrib")


def dp_count_by_unknown_keys_gaussian(df, column, epsilon, delta):
    """Counts per key when the key set is private, with Gaussian noise plus a threshold that hides
    rare keys. Replace-one neighbours (bounded DP, n public): d_in = 2. Result is (epsilon, delta)-DP.

    Returns ({key: count}, (epsilon_spent, delta_spent)).
    """
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    counts = space >> dp.t.then_count_by()
    l0, l1, li = counts.map(2)          # replace-one: keys touched, total change, change per key
    # The Gaussian threshold reads (l0, l2, linf); l2 <= min(l1, sqrt(l0)*li).
    d_mid = (l0, min(l1, l0 ** 0.5 * li), li)
    metric = dp.l02inf_distance(dp.absolute_distance(T=int))

    def make_g(scale, threshold):       # map(d_mid) -> (rho, delta of keys slipping past threshold)
        return dp.m.make_gaussian_threshold(counts.output_domain, metric, scale=scale, threshold=threshold)

    def make(scale, threshold, total_delta):   # rho -> privacy profile -> (eps, total_delta)
        return dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(make_g(scale, threshold)), delta=total_delta)

    # Half of delta for the Gaussian tail (scale), half for keys slipping past the threshold.
    scale = dp.binary_search(lambda s: make(s, 10**6, delta / 2).map(d_mid)[0] <= epsilon, bounds=(0.5, 1e4))
    threshold = dp.binary_search(lambda t: make_g(scale, t).map(d_mid)[1] <= delta / 2, bounds=(2, 10**9))   # threshold >= l-inf sensitivity (2 under replace-one)
    meas = make(scale, threshold, delta)
    epsilon_spent, delta_spent = meas.map(d_mid)
    assert epsilon_spent <= epsilon and delta_spent <= delta
    released = meas(counts(df[column].astype(str).tolist()))   # only keys whose noisy count passed
    return released, (epsilon_spent, delta_spent)


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 60 + ["b"] * 30 + ["c"]})
    _, (eps, _) = dp_count_by_unknown_keys_gaussian(df, "group", epsilon=1.0, delta=1e-6)
    return {"epsilon": float(eps)}
