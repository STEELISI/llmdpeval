import math

import numpy as np
from scipy.special import gammaln, xlog1py, xlogy


def _eps_add_remove(q, sigma, steps, delta, orders=range(2, 257)):
    """RDP of the Poisson-subsampled Gaussian (integer orders, Mironov-Talwar-Zhang 2019), composed over steps,
    converted to (eps, delta) with eps = rdp + log(1/delta)/(a-1) (Mironov 2017, Prop. 3). Add/remove-one."""
    best = math.inf
    for a in orders:
        k = np.arange(a + 1)
        terms = (gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1) + xlogy(k, q) + xlog1py(a - k, -q)
                 + (k * k - k) / (2 * sigma ** 2))
        best = min(best, steps * np.logaddexp.reduce(terms) / (a - 1) + math.log(1 / delta) / (a - 1))
    return best


def dp_sgd_mlp(df, feature_columns, label_column, positive_value, feature_bounds, epsilon, delta,
               hidden=16, epochs=5, expected_batch=256, clip=1.0, lr=0.5):
    """(eps, delta)-DP binary MLP classifier (one tanh hidden layer) by DP-SGD in NumPy: Poisson sampling,
    per-example clipping to L2 norm `clip`, Gaussian noise sigma*clip on the clipped sum (Abadi et al. 2016).
    Replace-one, N public. Returns ({"predict_proba", "sigma"}, epsilon_spent)."""
    rng = np.random.default_rng()
    fb = np.asarray(feature_bounds, dtype=float)                    # PUBLIC bounds -> scale to [-1, 1]
    cx, hx = fb.mean(axis=1), (fb[:, 1] - fb[:, 0]) / 2
    X = (np.clip(df[feature_columns].to_numpy(dtype=float), fb[:, 0], fb[:, 1]) - cx) / hx
    y = (df[label_column].to_numpy() == positive_value).astype(float)
    n, d = X.shape
    q, steps = expected_batch / n, int(epochs * n / expected_batch)
    # Replace-one = remove then add: (e, dl) add/remove gives (2e, dl*(1+exp(e))) replace-one. Target e = eps/2.
    eps_ar, delta_ar = epsilon / 2, delta / (1 + math.exp(epsilon / 2))
    lo, hi = 0.3, 50.0                                              # bisection: smallest sigma meeting eps_ar
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if _eps_add_remove(q, mid, steps, delta_ar) <= eps_ar else (mid, hi)
    sigma = hi
    W1, b1 = rng.normal(0, 1 / math.sqrt(d), (d, hidden)), np.zeros(hidden)
    w2, b2 = rng.normal(0, 1 / math.sqrt(hidden), hidden), 0.0
    for _ in range(steps):
        idx = np.flatnonzero(rng.random(n) < q)                     # Poisson sampling, rate q
        xb, yb = X[idx], y[idx]
        A = np.tanh(xb @ W1 + b1)
        dz = 1 / (1 + np.exp(-(A @ w2 + b2))) - yb                  # d loss / d logit, per example
        dZ1 = dz[:, None] * w2 * (1 - A ** 2)
        G = np.hstack([(xb[:, :, None] * dZ1[:, None, :]).reshape(len(idx), -1), dZ1, dz[:, None] * A, dz[:, None]])
        G *= np.minimum(1, clip / np.maximum(np.linalg.norm(G, axis=1), 1e-12))[:, None]   # per-example clip
        g = (G.sum(axis=0) + rng.normal(0, sigma * clip, G.shape[1])) / (q * n)  # public denominator
        W1 -= lr * g[:d * hidden].reshape(d, hidden)
        b1, w2, b2 = b1 - lr * g[d * hidden:d * hidden + hidden], w2 - lr * g[-hidden - 1:-1], b2 - lr * g[-1]
    eps_spent = 2 * _eps_add_remove(q, sigma, steps, delta_ar)
    assert eps_spent <= epsilon + 1e-9 and delta_ar * (1 + math.exp(eps_spent / 2)) <= delta + 1e-15

    def predict_proba(x_new):                                       # public inputs, same public transform
        xs = (np.clip(np.asarray(x_new, dtype=float), fb[:, 0], fb[:, 1]) - cx) / hx
        return 1 / (1 + np.exp(-(np.tanh(xs @ W1 + b1) @ w2 + b2)))
    return {"predict_proba": predict_proba, "sigma": sigma}, float(eps_spent)


def _test():
    import pandas as pd
    rng = np.random.default_rng()
    a, b = rng.uniform(0, 10, 4000), rng.uniform(-5, 5, 4000)
    df = pd.DataFrame({"a": a, "b": b, "label": np.where((a - 5) ** 2 + b ** 2 < 12, "yes", "no")})
    model, eps = dp_sgd_mlp(df, ["a", "b"], "label", "yes", [(0, 10), (-5, 5)], 1.0, 1e-5, epochs=3)
    assert model["predict_proba"]([[5.0, 0.0], [9.0, 4.0]]).shape == (2,)
    return {"epsilon": eps}
