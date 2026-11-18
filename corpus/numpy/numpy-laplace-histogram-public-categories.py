import numpy as np


def dp_histogram(df, column, categories, epsilon, relation="replace_one"):
    """DP count for every category in a PUBLIC list (fixed order, never taken from the data).
    relation: "replace_one" (bounded DP, n public; default) or "add_remove" (n private).

    Returns ({category: private_count}, epsilon_spent). The whole vector costs epsilon once.
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    counts = np.array([np.sum(x == c) for c in categories], dtype=float)   # zeros included
    # L1 sensitivity of the count vector: add/remove moves one bin by 1;
    # replace-one moves one bin -1 and another +1.
    sensitivity = {"add_remove": 1.0, "replace_one": 2.0}[relation]
    scale = sensitivity / epsilon
    noisy = counts + rng.laplace(loc=0.0, scale=scale, size=len(categories))
    noisy = np.clip(np.round(noisy), 0, None)       # post-processing
    epsilon_spent = sensitivity / scale
    return dict(zip(categories, noisy.astype(int).tolist())), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    hist, epsilon_spent = dp_histogram(df, "group", ["a", "b", "c", "d"], epsilon=1.0,
                                       relation="replace_one")
    assert set(hist) == {"a", "b", "c", "d"}
    return {"epsilon": float(epsilon_spent)}
