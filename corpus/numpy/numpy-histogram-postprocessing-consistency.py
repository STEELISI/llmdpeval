import numpy as np


def _project_to_simplex(v, total):
    """Least-squares projection of v onto {c >= 0, sum(c) = total} (sort-based)."""
    if total <= 0:                                   # empty simplex face: only all zeros remain
        return np.zeros_like(v, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - total
    j = np.arange(1, len(u) + 1)
    r = j[u - css / j > 0][-1]
    return np.maximum(v - css[r - 1] / r, 0.0)


def dp_consistent_histogram(df, column, categories, epsilon, public_total=None, relation="replace_one"):
    """eps-DP counts over a PUBLIC category list, made non-negative, integer and summing to a total.

    relation: "replace_one" (bounded DP, n public; default) or "add_remove" (n private).
    public_total: the public N to sum to; None -> sum to the noisy total (always allowed).
    Returns ({category: int count}, epsilon_spent).
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    counts = np.array([np.sum(x == c) for c in categories], dtype=float)
    assert relation == "replace_one" or public_total is None, "under add/remove N is private"
    l1 = 2.0 if relation == "replace_one" else 1.0   # replace-one: one bin -1, another +1
    scale = l1 / epsilon
    noisy = counts + rng.laplace(0.0, scale, size=len(categories))
    total = public_total if public_total is not None else max(noisy.sum(), 0.0)   # post-processing
    proj = _project_to_simplex(noisy, total)
    out = np.floor(proj)
    rest = int(round(total - out.sum()))             # hand leftover units to largest fractions
    out[np.argsort(proj - out)[::-1][:rest]] += 1
    epsilon_spent = l1 / scale
    assert epsilon_spent <= epsilon + 1e-9
    return dict(zip(categories, out.astype(int).tolist())), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    res, eps = dp_consistent_histogram(df, "group", ["a", "b", "c", "d"], 1.0, public_total=len(df))
    assert sum(res.values()) == len(df) and min(res.values()) >= 0
    return {"epsilon": eps}
