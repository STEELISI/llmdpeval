import numpy as np


def dp_kary_randomized_response(df, column, categories, epsilon):
    """Local eps-DP k-ary randomized response over a PUBLIC category list + frequency estimates.

    Values not in `categories` must be mapped to an explicit category in the list beforehand.
    Returns ({"reported": indices, "frequencies": {category: share}}, epsilon_spent).
    """
    rng = np.random.default_rng()
    k = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    true = np.array([index[v] for v in df[column].to_numpy()])
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)   # report the truth
    q = 1.0 / (np.exp(epsilon) + k - 1)               # report one specific other category
    other = rng.integers(0, k - 1, size=len(true))
    other = other + (other >= true)                   # uniform over the k - 1 OTHER categories
    reported = np.where(rng.random(len(true)) < p, true, other)
    n_v = np.bincount(reported, minlength=k) / len(true)
    freq = (n_v - q) / (p - q)                        # unbiased; post-processing
    epsilon_spent = float(np.log(p / q))
    assert epsilon_spent <= epsilon + 1e-9
    return {"reported": reported, "frequencies": dict(zip(categories, freq.tolist()))}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"color": ["red", "blue", "red", "green", "red"] * 100})
    res, eps = dp_kary_randomized_response(df, "color", ["blue", "green", "red"], epsilon=1.0)
    assert set(res["frequencies"]) == {"blue", "green", "red"}
    return {"epsilon": eps}
