import numpy as np


def dp_top_k_by_count(df, column, categories, k, epsilon, relation="replace_one"):
    """DP top-k categories by record count: one Laplace histogram over a PUBLIC list, then sort.
    categories: PUBLIC list of every possible value, declared before looking at the data (codebook,
    schema), in a fixed order; never df[column].unique() (which values occur is itself private).

    relation: "replace_one" (bounded DP, n public; default) or "add_remove".
    Returns ({"top": [...], "noisy_counts": {...}}, epsilon_spent). Sorting is post-processing.
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    counts = np.array([np.sum(x == c) for c in categories], dtype=float)   # zeros included
    # L1 sensitivity of the count vector: 1 (add/remove) or 2 (replace-one: -1 in one bin, +1 in another)
    l1 = 1.0 if relation == "add_remove" else 2.0
    scale = l1 / epsilon                     # does not depend on k
    noisy = counts + rng.laplace(0.0, scale, size=len(categories))
    order = np.argsort(-noisy)[:k]
    epsilon_spent = l1 / scale
    assert epsilon_spent <= epsilon + 1e-9
    return {"top": [categories[i] for i in order],
            "noisy_counts": dict(zip(categories, noisy.tolist()))}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a", "e", "b"] * 20})
    res, eps = dp_top_k_by_count(df, "group", ["a", "b", "c", "d", "e"], k=2, epsilon=1.0,
                                 relation="replace_one")
    assert len(res["top"]) == 2
    return {"epsilon": eps}
