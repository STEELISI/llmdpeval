import numpy as np


def design_matrix(df, numeric, categorical):
    """numeric {col: (low, high)} PUBLIC bounds -> [0, 1]; categorical {col: [PUBLIC levels]} -> one-hot
    (unknown values -> all zeros). Same columns for train and test. Each block adds <= 1 to the squared
    norm, so scaling by 1/sqrt(2d) plus an intercept column 1/sqrt(2) gives row norm <= 1."""
    blocks = [((df[c].astype(float).clip(lo, hi) - lo) / (hi - lo)).to_numpy()[:, None] for c, (lo, hi) in numeric.items()]
    blocks += [np.column_stack([(df[c] == lv).to_numpy(float) for lv in levels]) for c, levels in categorical.items()]
    d = len(numeric) + len(categorical)
    return np.column_stack([b / np.sqrt(2 * d) for b in blocks] + [np.full(len(df), 1 / np.sqrt(2))])


def dp_logistic_with_group_accuracy(train, test, label, numeric, categorical, group_column, groups,
                                    epsilon, model_share=0.8, lam=None):
    """Objective-perturbation logistic regression (CMS 2011, Alg. 2; row norm <= 1, labels 0/1) on numeric +
    one-hot features, then overall and per-group accuracy from ONE noisy (group, correct) histogram.
    Replace-one (bounded DP: train and test sizes public; group sizes private). Returns (dict, eps)."""
    rng = np.random.default_rng()
    X, y = design_matrix(train, numeric, categorical), 2 * train[label].to_numpy() - 1   # labels in {-1, 1}
    n, p = X.shape
    eps_m = model_share * epsilon
    lam = lam or 1.0 / n                            # public regulariser
    eps_p = eps_m - 2 * np.log(1 + 0.25 / (n * lam))  # slack for logistic loss curvature c = 1/4
    delta_ridge = 0.0
    if eps_p <= 0:                                 # CMS fallback: extra ridge, half the budget for noise
        delta_ridge, eps_p = 0.25 / (n * (np.exp(eps_m / 4) - 1)) - lam, eps_m / 2
    direction = rng.normal(size=p)
    b = direction / np.linalg.norm(direction) * rng.gamma(p, 2 / eps_p)   # ||b|| ~ Gamma(p, 2/eps')
    w, reg = np.zeros(p), lam + delta_ridge          # minimise mean logistic loss + b.w/n + reg/2 |w|^2
    for _ in range(50):                            # Newton steps: the objective is strongly convex
        s = 1 / (1 + np.exp(y * (X @ w)))
        grad = -(X * (y * s)[:, None]).mean(axis=0) + reg * w + b / n
        w -= np.linalg.solve((X * (s * (1 - s))[:, None]).T @ X / n + reg * np.eye(p), grad)
    correct = ((design_matrix(test, numeric, categorical) @ w > 0).astype(int) == test[label].to_numpy())
    g = test[group_column].map({v: i for i, v in enumerate(groups)}).fillna(len(groups)).to_numpy().astype(int)
    cells = np.bincount(2 * g + correct, minlength=2 * (len(groups) + 1)).astype(float)
    eps_eval = epsilon - eps_m
    scale = 2.0 / eps_eval                         # replace-one: a test row leaves one cell, enters another
    noisy = np.maximum(cells + rng.laplace(0, scale, cells.size), 0).reshape(-1, 2)
    accuracy = float(np.clip(noisy[:, 1].sum() / len(test), 0, 1))   # test size is public
    group_acc = {v: float(np.clip(noisy[i, 1] / max(noisy[i].sum(), 1.0), 0, 1)) for i, v in enumerate(groups)}
    return {"weights": w, "accuracy": accuracy, "group_accuracy": group_acc}, eps_m + 2.0 / scale


def _test():
    rng = np.random.default_rng()
    import pandas as pd
    m = 2000
    df = pd.DataFrame({"h": rng.uniform(0, 80, m), "k": rng.choice(["s", "t", "u"], m),
                       "grp": rng.choice(["g1", "g2"], m)})
    df["y"] = ((df["h"] > 40) ^ (rng.random(m) < 0.1)).astype(int)
    res, eps = dp_logistic_with_group_accuracy(df[::2], df[1::2], "y", {"h": (0, 100)}, {"k": ["s", "t", "u"]},
                                               "grp", ["g1", "g2"], epsilon=1.0)
    assert 0 <= res["accuracy"] <= 1 and set(res["group_accuracy"]) == {"g1", "g2"}
    return {"epsilon": float(eps)}
