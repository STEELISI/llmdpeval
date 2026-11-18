import numpy as np


def dp_noisy_max_gumbel(df, column, candidates, epsilon, monotone=False):
    """DP arg-max of per-candidate counts over a PUBLIC candidate list, Gumbel noise.

    argmax(score + Gumbel(scale)) samples exactly from the exponential mechanism with
    weights exp(score / scale). Returns (winner, epsilon_spent); the scores stay secret.
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    scores = np.array([np.sum(x == c) for c in candidates], dtype=float)   # zeros included
    sens = 1.0                 # one record changes any single count by at most 1
    # Exponential mechanism: scale 2*sens/eps. Only monotone scores (counts under
    # add/remove) may use sens/eps.
    factor = 1.0 if monotone else 2.0
    scale = factor * sens / epsilon
    noisy = scores + rng.gumbel(loc=0.0, scale=scale, size=len(candidates))
    winner = candidates[int(np.argmax(noisy))]
    epsilon_spent = factor * sens / scale
    assert epsilon_spent <= epsilon + 1e-9
    return winner, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    winner, eps = dp_noisy_max_gumbel(df, "group", ["a", "b", "c", "d"], epsilon=1.0)
    assert winner in {"a", "b", "c", "d"}
    return {"epsilon": eps}
