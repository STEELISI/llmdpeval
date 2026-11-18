import numpy as np


def dp_logistic_gd(df, feature_columns, label_column, positive_value, feature_bounds, epsilon, delta,
                   steps=30, clip=1.0, lr=1.0):
    """(eps, delta)-DP logistic regression by full-batch clipped noisy gradient descent.

    Replace-one relation with PUBLIC N; feature_bounds [(lo, hi)] PUBLIC.
    Returns ({"w", "predict_proba"}, epsilon_spent). zCDP accounting over `steps` Gaussian steps.
    """
    rng = np.random.default_rng()
    fb = np.asarray(feature_bounds, dtype=float)
    cx, hx = fb.mean(axis=1), (fb[:, 1] - fb[:, 0]) / 2
    X = (np.clip(df[feature_columns].to_numpy(dtype=float), fb[:, 0], fb[:, 1]) - cx) / hx
    Xa = np.column_stack([np.ones(len(X)), X])
    y = (df[label_column].to_numpy() == positive_value).astype(float)
    n, d1 = Xa.shape                                    # N public by design
    log_d = np.log(1 / delta)
    rho = (np.sqrt(log_d + epsilon) - np.sqrt(log_d)) ** 2
    sens = 2.0 * clip                                   # replace-one
    sigma = sens * np.sqrt(steps / (2 * rho))
    w = np.zeros(d1)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Xa @ w)))
        g = (p - y)[:, None] * Xa                       # per-record gradients
        g *= np.minimum(1.0, clip / np.maximum(np.linalg.norm(g, axis=1), 1e-12))[:, None]
        w -= lr * (g.sum(axis=0) + rng.normal(0.0, sigma, size=d1)) / n
    rho_spent = steps * sens ** 2 / (2 * sigma ** 2)
    epsilon_spent = rho_spent + 2 * np.sqrt(rho_spent * log_d)
    assert epsilon_spent <= epsilon + 1e-9

    def predict_proba(x_new):                           # public inputs, same transform
        xs = (np.asarray(x_new, dtype=float).reshape(-1, len(cx)) - cx) / hx
        return 1.0 / (1.0 + np.exp(-(w[0] + xs @ w[1:])))
    return {"w": w, "predict_proba": predict_proba}, float(epsilon_spent)


def _test():
    import pandas as pd
    t = np.linspace(0, 1, 600)
    df = pd.DataFrame({"a": 50 * t, "b": 10 * t[::-1], "label": np.where(t > 0.5, "pos", "neg")})
    res, eps = dp_logistic_gd(df, ["a", "b"], "label", "pos", [(0, 50), (0, 10)], 1.0, 1e-5)
    assert res["predict_proba"]([[10.0, 3.0]]).shape == (1,)
    return {"epsilon": eps}
