import numpy as np


def dp_top_k_peeling(df, column, candidates, k, epsilon, monotone=False):
    """DP ordered top-k (by record count) of a PUBLIC candidate list, by peeling.
    candidates: PUBLIC list of every possible value, declared before looking at the data (codebook,
    schema), in a fixed order; never df[column].unique() (which values occur is itself private).

    k rounds of Laplace report-noisy-max, each at epsilon / k, removing the winner each time.
    monotone=False (default) is right for replace-one (bounded DP, n public).
    Returns ([winners in order], epsilon_spent). Noisy scores are never released.
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    scores = np.array([np.sum(x == c) for c in candidates], dtype=float)   # zeros included
    sens = 1.0                                   # per-score sensitivity of a count
    eps_round = epsilon / k                      # basic sequential composition over k rounds
    factor = 1.0 if monotone else 2.0            # monotone only for counts under add/remove
    scale = factor * sens / eps_round
    remaining, chosen, spent = list(range(len(candidates))), [], 0.0
    for _ in range(k):
        noisy = scores[remaining] + rng.laplace(0.0, scale, size=len(remaining))   # fresh noise
        chosen.append(remaining.pop(int(np.argmax(noisy))))
        spent += factor * sens / scale
    assert spent <= epsilon + 1e-9
    return [candidates[i] for i in chosen], spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a", "e", "b"] * 20})
    top, eps = dp_top_k_peeling(df, "group", ["a", "b", "c", "d", "e"], k=3, epsilon=1.0)
    assert len(set(top)) == 3
    return {"epsilon": eps}
