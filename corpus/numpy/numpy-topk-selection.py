import numpy as np


def dp_top_k(df, column, candidates, k, epsilon, relation="replace_one"):
    """DP ordered top-k by record count of a PUBLIC candidate list, one-shot Gumbel noise.

    Equals k rounds of exponential-mechanism peeling (Durfee & Rogers 2019), each at epsilon / k.
    candidates: every possible value, declared before looking at the data (codebook, schema),
    in a fixed order; never df[column].unique().
    Returns ([top-k candidates in order], epsilon_spent). Scores are never released.
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    scores = np.array([np.sum(x == c) for c in candidates], dtype=float)   # zeros included
    monotone = relation == "add_remove"       # replace-one (default): counts are NOT monotone
    sens = 1.0                                # per-score sensitivity of a count
    eps_pick = epsilon / k                    # total budget split over the k picks
    factor = 1.0 if monotone else 2.0         # monotone: scale = Δ/ε_pick; otherwise 2Δ/ε_pick
    scale = factor * sens / eps_pick
    noisy = scores + rng.gumbel(loc=0.0, scale=scale, size=len(candidates))
    order = np.argsort(-noisy)[:k]
    epsilon_spent = k * (factor * sens / scale)   # NumPy has no accountant: k picks at ε_pick each
    return [candidates[i] for i in order], epsilon_spent


def _test():
    import pandas as pd
    CANDIDATES = ["a", "b", "c", "d", "e"]   # public list, declared up front
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a", "e", "b"] * 20})
    top, eps = dp_top_k(df, "group", CANDIDATES, k=3, epsilon=1.0)
    assert len(set(top)) == 3
    return {"epsilon": eps}
