import opendp.prelude as dp

dp.enable_features("contrib")


def dp_fraction_matching(indicators, n_public, epsilon):
    """DP share of rows with indicator 1 (support of a predicate, accuracy as pred == label), bounded DP.

    indicators: one 0/1 value per private row; n_public: the PUBLIC row count (must equal len(indicators)).
    Replace-one neighbours (n public). Returns (private_fraction, epsilon_spent).
    """
    assert len(indicators) == n_public, "n must be public and match the data"
    space = (dp.vector_domain(dp.atom_domain(T=int), size=n_public), dp.symmetric_distance())

    def make(scale):
        # the clamp enforces the (0, 1) bounds; OpenDP does not check declared bounds on its own
        return space >> dp.t.then_clamp(bounds=(0, 1)) >> dp.t.then_sum() >> dp.m.then_laplace(scale=scale)

    meas = make(dp.binary_search_param(make, d_in=2, d_out=epsilon))   # replace-one = distance 2
    epsilon_spent = meas.map(2)
    noisy_count = meas([int(v) for v in indicators])
    return min(max(noisy_count / n_public, 0.0), 1.0), epsilon_spent   # post-processing


def _test():
    import pandas as pd
    df = pd.DataFrame({"pred": [1, 0, 1, 1] * 50, "label": [1, 0, 0, 1] * 50})
    acc, eps = dp_fraction_matching((df["pred"] == df["label"]).astype(int).tolist(), len(df), epsilon=1.0)
    assert 0.0 <= acc <= 1.0
    return {"epsilon": float(eps)}
