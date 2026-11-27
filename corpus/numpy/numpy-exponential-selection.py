import numpy as np


def dp_select(df, column, candidates, epsilon):
    """DP choice of one candidate from a PUBLIC list, scored by its record count; exponential mechanism.

    Returns (chosen_candidate, epsilon_spent). Only the winner is released, not its score.
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    u = np.array([np.sum(x == c) for c in candidates], dtype=float)   # utility: count, zeros included
    sens_u = 1.0                             # a count moves by at most 1 under add/remove and replace-one
    # Keep the factor 2: it may be dropped only for monotone utilities (counts under add/remove).
    coef = epsilon / (2 * sens_u)           # the coefficient actually used in the logits
    logits = coef * u
    logits -= np.max(logits)                 # log space, avoids overflow
    p = np.exp(logits)
    p /= p.sum()
    idx = rng.choice(len(candidates), p=p)
    epsilon_spent = 2 * sens_u * coef        # sampling ∝ exp(c·u) costs 2·c·Δu
    assert epsilon_spent <= epsilon + 1e-12
    return candidates[idx], epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    choice, epsilon_spent = dp_select(df, "group", ["a", "b", "c", "d"], epsilon=1.0)
    assert choice in {"a", "b", "c", "d"}
    return {"epsilon": epsilon_spent}
