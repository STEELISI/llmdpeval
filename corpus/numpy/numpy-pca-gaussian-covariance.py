import numpy as np


def dp_pca(df, columns, bounds, n_components, epsilon, delta, relation="replace_one"):
    """(eps, delta)-DP principal directions: Gaussian noise on X^T X (Analyze Gauss), then eigh.

    bounds: [(lo, hi)] per column, PUBLIC. Returns ({"components", "explained"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    b = np.asarray(bounds, dtype=float)
    c, h = b.mean(axis=1), (b[:, 1] - b[:, 0]) / 2
    X = (np.clip(df[columns].to_numpy(dtype=float), b[:, 0], b[:, 1]) - c) / h
    X = X / np.sqrt(len(columns))                    # every row has norm <= 1
    A = X.T @ X
    sens = 2.0 if relation == "replace_one" else 1.0  # ||x x^T||_F <= 1; replace-one = remove + add
    log_d = np.log(1 / delta)
    rho = (np.sqrt(log_d + epsilon) - np.sqrt(log_d)) ** 2
    sigma = sens / np.sqrt(2 * rho)
    iu = np.triu_indices(len(columns))
    E = np.zeros_like(A)
    E[iu] = rng.normal(0.0, sigma, size=len(iu[0]))
    A_n = A + E + np.triu(E, 1).T                    # symmetric noise
    vals, vecs = np.linalg.eigh(A_n)                 # post-processing
    order = np.argsort(vals)[::-1][:n_components]
    explained = np.clip(vals[order], 0, None) / max(np.clip(vals, 0, None).sum(), 1e-12)
    rho_spent = sens ** 2 / (2 * sigma ** 2)
    epsilon_spent = rho_spent + 2 * np.sqrt(rho_spent * log_d)
    assert epsilon_spent <= epsilon + 1e-9
    return {"components": vecs[:, order].T, "explained": explained}, float(epsilon_spent)


def _test():
    import pandas as pd
    t = np.linspace(0, 1, 500)
    df = pd.DataFrame({"a": 10 * t, "b": 5 * t + 1, "c": np.cos(7 * t)})
    res, eps = dp_pca(df, ["a", "b", "c"], [(0, 10), (0, 6), (-1, 1)], 2, epsilon=1.0, delta=1e-5)
    assert res["components"].shape == (2, 3)
    return {"epsilon": eps}
