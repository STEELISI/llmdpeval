import numpy as np


def dp_linear_gd(df, feature_columns, label_column, feature_bounds, label_bounds, epsilon, delta,
                 steps=20, clip=1.0, lr=None):
    """(eps, delta)-DP linear regression by full-batch clipped noisy gradient descent.

    Replace-one relation with PUBLIC N; bounds are PUBLIC. Accounting: zCDP over `steps` Gaussian
    releases. Returns ({"beta" (scaled space), "predict"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    fb = np.asarray(feature_bounds, dtype=float)
    cx, hx = fb.mean(axis=1), (fb[:, 1] - fb[:, 0]) / 2
    cy, hy = sum(label_bounds) / 2, (label_bounds[1] - label_bounds[0]) / 2
    X = (np.clip(df[feature_columns].to_numpy(dtype=float), fb[:, 0], fb[:, 1]) - cx) / hx
    y = (np.clip(df[label_column].to_numpy(dtype=float), *label_bounds) - cy) / hy
    Xa = np.column_stack([np.ones(len(X)), X])
    n, d1 = Xa.shape                                  # N public by design (replace-one)
    lr = lr or 1.0 / d1
    log_d = np.log(1 / delta)
    rho = (np.sqrt(log_d + epsilon) - np.sqrt(log_d)) ** 2
    sens = 2.0 * clip                                 # replace-one: one clipped gradient swapped
    sigma = sens * np.sqrt(steps / (2 * rho))
    w = np.zeros(d1)
    for _ in range(steps):
        g = (Xa @ w - y)[:, None] * Xa                # per-record gradients
        g *= np.minimum(1.0, clip / np.maximum(np.linalg.norm(g, axis=1), 1e-12))[:, None]
        w -= lr * (g.sum(axis=0) + rng.normal(0.0, sigma, size=d1)) / n
    rho_spent = steps * sens ** 2 / (2 * sigma ** 2)
    epsilon_spent = rho_spent + 2 * np.sqrt(rho_spent * log_d)
    assert epsilon_spent <= epsilon + 1e-9

    def predict(x_new):                               # public inputs: same transform, no clamp
        xs = (np.asarray(x_new, dtype=float).reshape(-1, len(cx)) - cx) / hx
        return (w[0] + xs @ w[1:]) * hy + cy
    return {"beta": w, "predict": predict}, float(epsilon_spent)


def _test():
    import pandas as pd
    t = np.linspace(0, 1, 400)
    df = pd.DataFrame({"x1": 10 * t, "x2": 4 * t ** 2, "y": 1 + 5 * t})
    res, eps = dp_linear_gd(df, ["x1", "x2"], "y", [(0, 10), (0, 4)], (0, 8), 1.0, 1e-5)
    assert res["predict"]([[5.0, 1.0]]).shape == (1,)
    return {"epsilon": eps}
