import numpy as np


def dp_top_k_by_rate(df, column, candidates, k, epsilon, outcome_column, outcome_value):
    """Top-k public candidates by the RATE of rows with outcome_value, with pure DP, replace-one (bounded DP, n public).
    candidates: PUBLIC list of every possible value, declared before looking at the data (codebook,
    schema), in a fixed order; never df[column].unique() (which values occur is itself private).

    Two Laplace histograms over the PUBLIC candidate list: outcome rows per group (numerator) and all
    rows per group (denominator), epsilon/2 each. rate = noisy_num / noisy_den is post-processing.
    Returns ({"top": [...], "noisy_rates": {...}}, epsilon_spent).
    """
    rng = np.random.default_rng()
    groups = df[column].to_numpy()
    hit = (df[outcome_column] == outcome_value).to_numpy()
    num = np.array([np.sum(hit & (groups == c)) for c in candidates], dtype=float)
    den = np.array([np.sum(groups == c) for c in candidates], dtype=float)
    # Replace-one: a changed row leaves one bin and enters another, L1 sensitivity 2 each (1 under add/remove).
    l1 = 2.0
    scale = l1 / (epsilon / 2)               # two releases, epsilon/2 each (sequential composition)
    noisy_num = num + rng.laplace(0.0, scale, size=len(candidates))
    noisy_den = den + rng.laplace(0.0, scale, size=len(candidates))
    # Post-processing: ignore groups whose NOISY size is below 10 noise scales (a rate of a tiny
    # group is mostly noise). The rule is fixed in advance and uses only released values.
    keep = noisy_den >= 10 * scale
    rate = np.where(keep, noisy_num / np.maximum(noisy_den, 1.0), -np.inf)
    order = [i for i in np.argsort(-rate) if np.isfinite(rate[i])][:k]   # groups below the cut-off are never picked
    epsilon_spent = 2 * (l1 / scale)
    assert epsilon_spent <= epsilon + 1e-9
    return {"top": [candidates[i] for i in order],
            "noisy_rates": dict(zip(candidates, rate.tolist()))}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 300 + ["b"] * 300 + ["c"] * 300 + ["d"] * 3,
                       "flag": (["y"] * 240 + ["n"] * 60) + (["y"] * 90 + ["n"] * 210)
                               + (["y"] * 150 + ["n"] * 150) + ["y"] * 3})
    res, eps = dp_top_k_by_rate(df, "group", ["a", "b", "c", "d"], k=2, epsilon=1.0,
                                outcome_column="flag", outcome_value="y")
    assert len(res["top"]) == 2
    return {"epsilon": eps}
