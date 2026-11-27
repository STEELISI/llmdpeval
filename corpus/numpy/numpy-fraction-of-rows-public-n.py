import numpy as np


def dp_fraction_matching(indicators, n_public, epsilon):
    """DP share of rows with indicator 1 (support of a predicate, accuracy as pred == label).

    Replace-one neighbours with a PUBLIC n: the count has sensitivity 1.
    Returns (private_fraction, epsilon_spent).
    """
    x = np.asarray(indicators, dtype=float)
    assert len(x) == n_public and set(np.unique(x)) <= {0.0, 1.0}
    rng = np.random.default_rng()
    sensitivity = 1.0                       # one replaced row flips at most one indicator
    scale = sensitivity / epsilon
    noisy = x.sum() + rng.laplace(0.0, scale)
    epsilon_spent = sensitivity / scale     # NumPy: implied by sensitivity and scale
    return float(np.clip(noisy / n_public, 0.0, 1.0)), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"pred": [1, 0, 1, 1] * 50, "label": [1, 0, 0, 1] * 50})
    acc, eps = dp_fraction_matching((df["pred"] == df["label"]).astype(int), len(df), epsilon=1.0)
    assert 0.0 <= acc <= 1.0
    return {"epsilon": float(eps)}
