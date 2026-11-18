import numpy as np


def dp_consistent_marginals(df, marginals, categories, n, epsilon, iters=100):
    """Noisy 1-way and 2-way marginals under one epsilon, then post-processed to be non-negative,
    sum to the PUBLIC n, and agree on every shared 1-way margin. Replace-one (bounded DP, n public).

    marginals: list of column tuples of length 1 or 2; categories: {column: public list}.
    Returns ({tuple: ndarray}, epsilon_spent).
    """
    rng = np.random.default_rng()
    m = len(marginals)
    scale = 2 * m / epsilon        # replace-one: one row leaves a cell and enters another in EACH table, L1 = 2m
    noisy = {}
    for t in marginals:
        idx = [df[c].map({v: i for i, v in enumerate(categories[c])}).to_numpy() for c in t]
        keep = ~np.any([np.isnan(i.astype(float)) for i in idx], axis=0)   # outside the public lists: dropped
        shape = [len(categories[c]) for c in t]
        exact = np.bincount(np.ravel_multi_index([i[keep].astype(int) for i in idx], shape),
                            minlength=int(np.prod(shape))).reshape(shape)
        noisy[t] = exact + rng.laplace(0.0, scale, size=shape)
    # Post-processing (no raw data from here on). 1) Each column's 1-way target = inverse-variance average of
    # every noisy estimate of it (a margin summed over K cells has K times the variance), clipped, rescaled to n.
    target = {}
    for c in categories:
        est = [(noisy[t].sum(axis=1 - t.index(c)) if len(t) == 2 else noisy[t],
                1.0 / (len(categories[t[1 - t.index(c)]]) if len(t) == 2 else 1)) for t in noisy if c in t]
        if est:
            avg = np.maximum(sum(w * e for e, w in est) / sum(w for _, w in est), 0.0)
            target[c] = avg * n / avg.sum() if avg.sum() > 0 else np.full(avg.shape, n / avg.size)
    # 2) Each 2-way table: clip at a small floor, then iterative proportional fitting onto both targets.
    out = {(c,): target[c] for c in target}
    for t in (t for t in noisy if len(t) == 2):
        tab = np.maximum(noisy[t], 1e-3)
        for _ in range(iters):
            tab *= (target[t[0]] / np.maximum(tab.sum(axis=1), 1e-12))[:, None]
            tab *= (target[t[1]] / np.maximum(tab.sum(axis=0), 1e-12))[None, :]
        out[t] = tab
    epsilon_spent = 2 * m / scale
    assert epsilon_spent <= epsilon + 1e-9
    return {t: out[t] for t in marginals}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"a": ["x", "y", "x", "z"] * 50, "b": ["u", "v", "v", "u"] * 50, "c": ["s", "t"] * 100})
    cats = {"a": ["x", "y", "z", "w"], "b": ["u", "v"], "c": ["s", "t"]}
    res, eps = dp_consistent_marginals(df, [("a",), ("b",), ("a", "b"), ("b", "c")], cats, 200, epsilon=1.0)
    assert np.allclose(res[("a", "b")].sum(axis=0), res[("b", "c")].sum(axis=1), atol=1e-6)
    assert min(v.min() for v in res.values()) >= 0
    return {"epsilon": eps}
