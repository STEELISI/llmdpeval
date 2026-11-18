import numpy as np


def dp_ssp_centred(df, feature_columns, label_column, feature_bounds, label_bounds, epsilon,
                   x_new, shares=(0.2, 0.2, 0.3, 0.3), ridge=1e-3):
    """Pure eps-DP least squares from four Laplace-noised sums, centred on PUBLIC midpoints.

    Replace-one relation, N public. feature_bounds [(lo, hi)] and label_bounds (lo, hi) PUBLIC.
    shares: epsilon split over (sum x, sum y, X^T X upper triangle, X^T y).
    Returns ({"slope", "intercept", "predictions"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    fb = np.asarray(feature_bounds, dtype=float)
    cx, hx = fb.mean(axis=1), (fb[:, 1] - fb[:, 0]) / 2       # public midpoints / half-widths
    cy, hy = sum(label_bounds) / 2, (label_bounds[1] - label_bounds[0]) / 2
    X = np.clip(df[feature_columns].to_numpy(dtype=float), fb[:, 0], fb[:, 1]) - cx   # in [-h, h]
    y = np.clip(df[label_column].to_numpy(dtype=float), *label_bounds) - cy
    n, d = X.shape                                            # N public by design
    iu = np.triu_indices(d)
    hh = np.outer(hx, hx)
    sens_xx = np.where(iu[0] == iu[1], hh[iu], 2 * hh[iu]).sum()   # diag in [0, h^2], off-diag +-h h
    sens = np.array([2 * hx.sum(), 2 * hy, sens_xx, 2 * hy * hx.sum()])   # replace-one L1
    eps_i = epsilon * np.asarray(shares)
    scale = sens / eps_i
    s_x = X.sum(axis=0) + rng.laplace(0.0, scale[0], size=d)
    s_y = y.sum() + rng.laplace(0.0, scale[1])
    xx = (X.T @ X)[iu] + rng.laplace(0.0, scale[2], size=len(iu[0]))
    xy = X.T @ y + rng.laplace(0.0, scale[3], size=d)
    A = np.zeros((d, d))                                      # post-processing from here on
    A[iu] = xx
    A = A + np.triu(A, 1).T
    m_x, m_y = s_x / n, s_y / n
    C = A / n - np.outer(m_x, m_x)
    slope = np.linalg.solve(C + ridge * np.eye(d), xy / n - m_x * m_y)
    intercept = m_y - m_x @ slope
    xs = np.asarray(x_new, dtype=float).reshape(-1, d) - cx   # same shift, no clipping
    preds = intercept + xs @ slope + cy
    epsilon_spent = float(np.sum(sens / scale))               # sequential composition
    assert epsilon_spent <= epsilon + 1e-9
    return {"slope": slope, "intercept": intercept + cy - cx @ slope, "predictions": preds}, epsilon_spent


def _test():
    import pandas as pd
    t = np.linspace(0, 1, 400)
    df = pd.DataFrame({"x1": 100 + 50 * t, "x2": 3 + 2 * t ** 2, "y": 10 + 20 * t})
    res, eps = dp_ssp_centred(df, ["x1", "x2"], "y", [(100, 150), (3, 5)], (10, 30), 1.0,
                              x_new=[[120.0, 4.0]])
    assert res["predictions"].shape == (1,)
    return {"epsilon": eps}
