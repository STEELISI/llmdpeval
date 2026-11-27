import numpy as np


def dp_noisy_argmax(df, candidates, score_fn, sensitivity, epsilon, monotone=False):
    """Report noisy max over a PUBLIC candidate list for any score function.

    score_fn(df, candidate) -> float. sensitivity: largest change of ONE candidate's score
    when one record changes (state the relation). Returns (winner, epsilon_spent).
    """
    rng = np.random.default_rng()
    scores = np.array([score_fn(df, c) for c in candidates], dtype=float)
    factor = 1.0 if monotone else 2.0          # Laplace RNM: Δ/ε only for monotone scores
    scale = factor * sensitivity / epsilon
    noisy = scores + rng.laplace(0.0, scale, size=len(candidates))
    epsilon_spent = factor * sensitivity / scale
    assert epsilon_spent <= epsilon + 1e-9
    return candidates[int(np.argmax(noisy))], epsilon_spent


def group_mean_score(value_column, group_column, lower, upper):
    """Score = mean of a column clamped to PUBLIC [lower, upper] inside one group.

    Sensitivity = upper - lower: a group may hold a single record. An empty group scores
    `lower`, so adding its first record also changes the score by at most upper - lower.
    A rate is the special case lower=0, upper=1 on a 0/1 column (sensitivity 1).
    """
    def score(df, c):
        v = np.clip(df.loc[df[group_column] == c, value_column].to_numpy(dtype=float), lower, upper)
        return float(v.mean()) if len(v) else float(lower)
    return score


def _test():
    import pandas as pd
    df = pd.DataFrame({"g": ["a", "b", "a", "c"] * 25, "v": [1.0, 0.0, 1.0, 0.5] * 25})
    winner, eps = dp_noisy_argmax(df, ["a", "b", "c", "d"], group_mean_score("v", "g", 0.0, 1.0),
                                  sensitivity=1.0, epsilon=1.0)
    assert winner in {"a", "b", "c", "d"}
    return {"epsilon": eps}
