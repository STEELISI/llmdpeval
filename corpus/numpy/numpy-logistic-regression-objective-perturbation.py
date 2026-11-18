import numpy as np


def dp_logistic_objective_perturbation(df, feature_columns, label_column, positive_value,
                                       feature_bounds, epsilon, lam=0.01, iters=50):
    """Pure eps-DP logistic regression, objective perturbation (Chaudhuri et al. 2011, Alg. 2).

    Replace-one relation, N public; PUBLIC feature bounds and lam. Rows scaled to ||x|| <= 1.
    Returns (w_private, epsilon_spent).
    """
    rng = np.random.default_rng()
    fb = np.asarray(feature_bounds, dtype=float)
    cx, hx = fb.mean(axis=1), (fb[:, 1] - fb[:, 0]) / 2
    X = (np.clip(df[feature_columns].to_numpy(dtype=float), fb[:, 0], fb[:, 1]) - cx) / hx
    X = np.column_stack([np.ones(len(X)), X]) / np.sqrt(len(cx) + 1)   # ||x|| <= 1 exactly
    y = np.where(df[label_column].to_numpy() == positive_value, 1.0, -1.0)
    n, d = X.shape
    c = 0.25                                                 # |loss''| <= 1/4 for logistic loss
    eps_p = epsilon - np.log(1 + 2 * c / (n * lam) + c ** 2 / (n * lam) ** 2)
    extra = 0.0
    if eps_p <= 0:
        extra = c / (n * (np.exp(epsilon / 4) - 1)) - lam
        eps_p = epsilon / 2
    z = rng.normal(size=d)
    b = rng.gamma(shape=d, scale=2.0 / eps_p) * z / np.linalg.norm(z)   # density ∝ exp(-eps_p ||b|| / 2)
    reg = lam + extra
    w = np.zeros(d)
    for _ in range(iters):                                   # Newton: exact minimiser
        s = 1.0 / (1.0 + np.exp(y * (X @ w)))
        grad = -(X * (y * s)[:, None]).mean(axis=0) + reg * w + b / n
        H = (X.T * (s * (1 - s))) @ X / n + reg * np.eye(d)
        w -= np.linalg.solve(H, grad)
    epsilon_spent = eps_p + 2 * np.log(1 + c / (n * reg))
    assert epsilon_spent <= epsilon + 1e-9
    return w, float(epsilon_spent)


def _test():
    import pandas as pd
    t = np.linspace(0, 1, 800)
    df = pd.DataFrame({"a": 50 * t, "b": 10 * t[::-1], "label": np.where(t > 0.5, "pos", "neg")})
    w, eps = dp_logistic_objective_perturbation(df, ["a", "b"], "label", "pos", [(0, 50), (0, 10)], 1.0)
    assert w.shape == (3,)
    return {"epsilon": eps}
