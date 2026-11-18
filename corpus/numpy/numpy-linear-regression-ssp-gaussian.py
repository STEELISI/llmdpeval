import numpy as np


def dp_ssp_regression(df, feature_columns, label_column, feature_bounds, label_bounds,
                      epsilon, delta, x_new, relation="replace_one"):
    """(eps, delta)-DP least squares: Gaussian noisy X^T X, X^T y and lambda_min (AdaSSP-style ridge).

    feature_bounds [(lo, hi)] and label_bounds (lo, hi) are PUBLIC. x_new: public inputs.
    Returns ({"beta", "predictions"}, epsilon_spent). Accounting through zCDP.
    """
    rng = np.random.default_rng()
    fb = np.asarray(feature_bounds, dtype=float)
    cx, hx = fb.mean(axis=1), (fb[:, 1] - fb[:, 0]) / 2
    cy, hy = sum(label_bounds) / 2, (label_bounds[1] - label_bounds[0]) / 2
    X = (np.clip(df[feature_columns].to_numpy(dtype=float), fb[:, 0], fb[:, 1]) - cx) / hx  # [-1, 1]
    y = (np.clip(df[label_column].to_numpy(dtype=float), *label_bounds) - cy) / hy           # [-1, 1]
    Xa = np.column_stack([np.ones(len(X)), X])
    d1 = Xa.shape[1]
    b2 = float(d1)                                   # ||row||^2 <= 1 + D
    log_d = np.log(1 / delta)
    rho = (np.sqrt(log_d + epsilon) - np.sqrt(log_d)) ** 2   # (eps, delta) -> rho-zCDP budget
    factor = 2.0 if relation == "replace_one" else 1.0      # replace-one (default) = remove + add
    sens = factor * np.array([b2, b2, np.sqrt(b2)])  # lambda_min, triu(X^T X), X^T y (L2)
    sig = sens / np.sqrt(2 * rho / 3)                # rho / 3 each
    A, v = Xa.T @ Xa, Xa.T @ y
    lam_min = np.linalg.eigvalsh(A)[0] + rng.normal(0.0, sig[0])
    iu = np.triu_indices(d1)
    E = np.zeros((d1, d1))
    E[iu] = rng.normal(0.0, sig[1], size=len(iu[0]))
    A_n = A + E + np.triu(E, 1).T                    # symmetric noise
    v_n = v + rng.normal(0.0, sig[2], size=d1)
    ridge = max(0.0, 2 * sig[1] * np.sqrt(d1) - lam_min) + sig[1]   # post-processing
    beta = np.linalg.solve(A_n + ridge * np.eye(d1), v_n)
    xs = (np.asarray(x_new, dtype=float).reshape(-1, len(cx)) - cx) / hx   # same transform, no clamp
    preds = (beta[0] + xs @ beta[1:]) * hy + cy
    rho_spent = float(np.sum(sens ** 2 / (2 * sig ** 2)))
    epsilon_spent = rho_spent + 2 * np.sqrt(rho_spent * log_d)
    assert epsilon_spent <= epsilon + 1e-9
    return {"beta": beta, "predictions": preds}, float(epsilon_spent)


def _test():
    import pandas as pd
    t = np.linspace(0, 1, 500)
    df = pd.DataFrame({"x1": 10 * t, "x2": 5 * t[::-1], "y": 2 + 3 * t})
    res, eps = dp_ssp_regression(df, ["x1", "x2"], "y", [(0, 10), (0, 5)], (0, 10), 1.0, 1e-5,
                                 x_new=[[1.0, 2.0], [8.0, 1.0]])
    assert res["predictions"].shape == (2,)
    return {"epsilon": eps}
