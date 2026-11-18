import numpy as np


def dp_permute_and_flip(df, column, candidates, epsilon):
    """DP choice of one candidate from a PUBLIC list scored by record count; permute-and-flip.

    McKenna & Sheldon 2020: visit candidates in random order, accept r with probability
    exp(eps * (q_r - q_max) / (2 * sens)). Returns (choice, epsilon_spent).
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    q = np.array([np.sum(x == c) for c in candidates], dtype=float)   # zeros included
    sens = 1.0                       # a count changes by at most 1 per record
    rate = epsilon / (2 * sens)
    q_star = q.max()                 # internal only, never released
    for r in rng.permutation(len(candidates)):
        if rng.random() <= np.exp(rate * (q[r] - q_star)):
            return candidates[r], 2 * sens * rate
    raise RuntimeError("unreachable: the best candidate is accepted with probability 1")


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    choice, eps = dp_permute_and_flip(df, "group", ["a", "b", "c", "d"], epsilon=1.0)
    assert choice in {"a", "b", "c", "d"}
    return {"epsilon": eps}
