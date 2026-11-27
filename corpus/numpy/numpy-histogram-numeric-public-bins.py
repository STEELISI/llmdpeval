import numpy as np


def dp_numeric_histogram(df, column, edges, epsilon, relation="replace_one", clamp=True):
    """DP histogram of a numeric column over PUBLIC bin edges, Laplace noise.

    clamp=True puts out-of-range values in the edge bins; False drops them (both per-record).
    relation: "replace_one" (bounded DP, n public; default) or "add_remove".
    Returns ({"edges", "counts"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    e = np.asarray(edges, dtype=float)
    x = df[column].to_numpy(dtype=float)
    if clamp:
        x = np.clip(x, e[0], e[-1])
    counts = np.histogram(x, bins=e)[0].astype(float)     # explicit edges, never bins=<int>
    # L1 sensitivity: 1 (add/remove), 2 (replace-one: one bin -1, another +1)
    l1 = 1.0 if relation == "add_remove" else 2.0
    scale = l1 / epsilon
    noisy = counts + rng.laplace(0.0, scale, size=len(counts))
    noisy = np.clip(np.rint(noisy), 0, None)              # post-processing
    epsilon_spent = l1 / scale
    assert epsilon_spent <= epsilon + 1e-9
    return {"edges": e.tolist(), "counts": noisy.tolist()}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": (np.arange(200) % 45 - 5).astype(float)})
    res, eps = dp_numeric_histogram(df, "value", np.linspace(0, 40, 9), epsilon=1.0, relation="replace_one")
    assert len(res["counts"]) == 8
    return {"epsilon": eps}
