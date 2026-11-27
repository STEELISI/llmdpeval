import itertools

import numpy as np
import pandas as pd


def dp_synthetic_rows(df, columns, categories, n_synth, epsilon, edges=None):
    """Synthetic categorical rows from a tree model fitted ONLY to noisy marginals (replace-one: bounded DP, n public).

    edges: public tree as [(parent, child), ...] chosen without the data; [] = independent columns;
    None = learn a Chow-Liu tree: release ALL pairs, then pick the maximum-mutual-information spanning tree
    from the noisy tables (post-processing, so the selection costs no extra epsilon).
    Returns (synthetic DataFrame, epsilon_spent).
    """
    rng = np.random.default_rng()
    pairs = list(itertools.combinations(columns, 2)) if edges is None else list(edges)
    roots = [columns[0]] if edges is None else [c for c in columns if c not in {ch for _, ch in edges}]
    tables = [(c,) for c in roots] + pairs   # 1-way tables only for roots: fewer tables, less noise per table
    scale = 2 * len(tables) / epsilon   # replace-one: a changed row moves 2 cells in EACH table, L1 = 2 * tables
    codes = {c: df[c].map({v: i for i, v in enumerate(categories[c])}).fillna(-1).astype(int).to_numpy()
             for c in columns}          # values outside the public lists are dropped from every table
    p = {}
    for t in tables:
        shape = [len(categories[c]) for c in t]
        ok = np.all([codes[c] >= 0 for c in t], axis=0)
        exact = np.bincount(np.ravel_multi_index([codes[c][ok] for c in t], shape),
                            minlength=int(np.prod(shape))).reshape(shape)
        noisy = np.maximum(exact + rng.laplace(0.0, scale, size=shape), 0.0) + 1e-6   # clip: post-processing
        p[t] = noisy / noisy.sum()
    if edges is None:   # Prim's maximum spanning tree on mutual information of the noisy pair tables
        mi = {t: float((p[t] * np.log(p[t] / np.outer(p[t].sum(1), p[t].sum(0)))).sum()) for t in pairs}
        edges, seen = [], {columns[0]}
        while len(seen) < len(columns):
            a, b = max((t for t in pairs if (t[0] in seen) != (t[1] in seen)), key=mi.get)
            edges.append((a, b) if a in seen else (b, a))
            seen |= {a, b}
    out = {}
    for c in roots:                  # roots: sample from the noisy 1-way margin
        out[c] = rng.choice(len(categories[c]), size=n_synth, p=p[(c,)])
    for parent, child in edges:      # edges in sampling order, parent first (Prim's order already is)
        joint = p[(parent, child)] if (parent, child) in p else p[(child, parent)].T
        cond = joint / joint.sum(axis=1, keepdims=True)
        u = rng.random(n_synth)[:, None]
        out[child] = np.minimum((u > cond[out[parent]].cumsum(axis=1)).sum(axis=1), len(categories[child]) - 1)
    epsilon_spent = 2 * len(tables) / scale
    assert epsilon_spent <= epsilon + 1e-9
    return pd.DataFrame({c: np.asarray(categories[c], dtype=object)[out[c]] for c in columns}), epsilon_spent


def _test():
    df = pd.DataFrame({"a": ["x", "y", "x", "z"] * 50, "b": ["u", "v", "v", "u"] * 50, "c": ["s", "t"] * 100})
    cats = {"a": ["x", "y", "z", "w"], "b": ["u", "v"], "c": ["s", "t"]}
    syn, eps = dp_synthetic_rows(df, ["a", "b", "c"], cats, 300, epsilon=1.0)
    assert syn.shape == (300, 3) and set(syn["a"]) <= set(cats["a"])
    _, eps2 = dp_synthetic_rows(df, ["a", "b", "c"], cats, 50, epsilon=1.0, edges=[("a", "b"), ("b", "c")])
    return {"epsilon": max(eps, eps2)}
