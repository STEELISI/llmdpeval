import numpy as np


def dp_noisy_max(df, column, candidates, epsilon, relation="replace_one"):
    """DP index of the most common candidate in a PUBLIC list, Report Noisy Max with Laplace noise.

    relation: "replace_one" (bounded DP, n public; default) or "add_remove".
    Returns (winning_candidate, epsilon_spent). The noisy scores stay secret.
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    c = np.array([np.sum(x == k) for k in candidates], dtype=float)   # zeros included
    sens = 1.0                                   # per-score sensitivity of a count
    # Counts under add/remove are monotone: scale = Δ/ε. Under replace-one one count falls
    # while another rises (not monotone): scale = 2Δ/ε.
    factor = 1.0 if relation == "add_remove" else 2.0
    scale = factor * sens / epsilon
    noisy = c + rng.laplace(loc=0.0, scale=scale, size=len(candidates))
    idx = int(np.argmax(noisy))                  # release only the arg-max
    epsilon_spent = factor * sens / scale
    return candidates[idx], epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    winner, epsilon_spent = dp_noisy_max(df, "group", ["a", "b", "c", "d"], epsilon=1.0,
                                         relation="replace_one")
    assert winner in {"a", "b", "c", "d"}
    return {"epsilon": epsilon_spent}
