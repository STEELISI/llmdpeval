import numpy as np


def _fit_logreg(X, y, lam, iters=50):
    """Exact minimiser of mean log-loss + lam/2 ||w||^2 by Newton's method."""
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(iters):
        s = 1.0 / (1.0 + np.exp(y * (X @ w)))               # sigmoid(-y w.x)
        grad = -(X * (y * s)[:, None]).mean(axis=0) + lam * w
        H = (X.T * (s * (1 - s))) @ X / n + lam * np.eye(d)
        w -= np.linalg.solve(H, grad)
    return w


def dp_logistic_output_perturbation(df, feature_columns, label_column, positive_value,
                                    feature_bounds, epsilon, lam=0.05):
    """Pure eps-DP logistic regression, output perturbation (Chaudhuri et al. 2011, Alg. 1).

    Replace-one relation, N public, PUBLIC feature bounds and lam. Returns (w_private, epsilon_spent).
    """
    rng = np.random.default_rng()
    fb = np.asarray(feature_bounds, dtype=float)
    cx, hx = fb.mean(axis=1), (fb[:, 1] - fb[:, 0]) / 2
    X = (np.clip(df[feature_columns].to_numpy(dtype=float), fb[:, 0], fb[:, 1]) - cx) / hx
    X = np.column_stack([np.ones(len(X)), X]) / np.sqrt(len(cx) + 1)   # ||x|| <= 1 exactly
    y = np.where(df[label_column].to_numpy() == positive_value, 1.0, -1.0)
    n, d = X.shape
    w = _fit_logreg(X, y, lam)
    sens = 2.0 / (n * lam)                                  # L2 sensitivity of the minimiser
    gamma_scale = sens / epsilon                            # density ∝ exp(-eps ||b|| / sens)
    z = rng.normal(size=d)
    b = rng.gamma(shape=d, scale=gamma_scale) * z / np.linalg.norm(z)
    epsilon_spent = sens / gamma_scale
    assert epsilon_spent <= epsilon + 1e-9
    return w + b, epsilon_spent     # predict with the same scaling: sigmoid(w . [1, xs] / sqrt(D+1))


def _test():
    import pandas as pd
    t = np.linspace(0, 1, 800)
    df = pd.DataFrame({"a": 50 * t, "b": 10 * t[::-1], "label": np.where(t > 0.5, "pos", "neg")})
    w, eps = dp_logistic_output_perturbation(df, ["a", "b"], "label", "pos", [(0, 50), (0, 10)], 1.0)
    assert w.shape == (3,)
    return {"epsilon": eps}
