import numpy as np


def dp_linear_regression(df, feature_columns, label_column, feature_bounds, label_bounds,
                         epsilon, x_new, ridge=1.0, relation="replace_one", share_a=0.5):
    """DP least squares by sufficient statistics perturbation: noisy X^T X and X^T y, then solve.

    feature_bounds: [(lower, upper)] per feature; label_bounds: (lower, upper); all public.
    x_new: public inputs to predict at (shifted, NOT clamped). Returns ({beta, predictions}, eps).
    """
    rng = np.random.default_rng()
    fb = np.asarray(feature_bounds, dtype=float)
    cx, bx = fb.mean(axis=1), (fb[:, 1] - fb[:, 0]) / 2          # centre on public midpoints
    cy, by = sum(label_bounds) / 2, (label_bounds[1] - label_bounds[0]) / 2
    X = np.clip(df[feature_columns].to_numpy(dtype=float) - cx, -bx, bx)
    y = np.clip(df[label_column].to_numpy(dtype=float) - cy, -by, by)
    Xa = np.column_stack([np.ones(len(X)), X])                   # intercept column, bound 1
    ba = np.concatenate([[1.0], bx])
    d1 = Xa.shape[1]
    iu = np.triu_indices(d1)
    a, v = (Xa.T @ Xa)[iu], Xa.T @ y                             # release upper triangle only
    # L1 sensitivity: one row adds x x^T and x y; replace-one (default) = remove + add, so twice that.
    factor = 2.0 if relation == "replace_one" else 1.0
    sens_a = factor * np.sum(np.outer(ba, ba)[iu])
    sens_v = factor * by * np.sum(ba)
    scale_a, scale_v = sens_a / (share_a * epsilon), sens_v / ((1 - share_a) * epsilon)
    a_n = a + rng.laplace(0.0, scale_a, size=len(a))
    v_n = v + rng.laplace(0.0, scale_v, size=d1)
    A_n = np.zeros((d1, d1))                                     # post-processing from here on
    A_n[iu] = a_n
    A_n = A_n + np.triu(A_n, 1).T
    beta = np.linalg.solve(A_n + ridge * np.eye(d1), v_n)        # public ridge: A_n may be singular
    xn = np.asarray(x_new, dtype=float).reshape(-1, len(cx)) - cx   # same shift, no clamping
    preds = beta[0] + xn @ beta[1:] + cy
    epsilon_spent = sens_a / scale_a + sens_v / scale_v          # sequential composition
    assert epsilon_spent <= epsilon + 1e-9
    return {"beta": beta, "predictions": preds}, epsilon_spent


def _test():
    import pandas as pd
    t = np.linspace(0, 1, 200)
    df = pd.DataFrame({"x1": 10 * t, "y": 3 + 2 * (10 * t)})
    res, epsilon_spent = dp_linear_regression(df, ["x1"], "y", [(0.0, 10.0)], (0.0, 30.0),
                                              epsilon=1.0, x_new=[[2.0], [15.0]])
    assert res["predictions"].shape == (2,)
    return {"epsilon": epsilon_spent}
