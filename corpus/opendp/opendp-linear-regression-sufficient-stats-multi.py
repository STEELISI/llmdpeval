import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


def dp_ols_multi(df, feature_columns, target_column, feature_bounds, target_bounds, epsilon):
    """OLS y ~ features + intercept from noisy X^T X and X^T y, each entry an OpenDP bounded sum.

    Replace-one (n public): sized domain, d_in = 2, each entry charged hi - lo; the intercept entry
    A[0, 0] = n is public and not released. (Add/remove-one, n private: unsized, d_in = 1, release it too.)

    feature_bounds: PUBLIC (lo, hi) per feature; target_bounds: PUBLIC (lo, hi).
    Returns (coef, epsilon_spent) with coef = [intercept, b_1..b_d].
    """
    cols = [np.ones(len(df))] + [df[c].to_numpy(float).clip(*b) for c, b in zip(feature_columns, feature_bounds)]
    cols.append(df[target_column].to_numpy(float).clip(*target_bounds))
    bounds = [(1.0, 1.0)] + list(feature_bounds) + [target_bounds]
    p = len(cols) - 1                                   # intercept + d features
    pairs = [(i, j) for i in range(p + 1) for j in range(i, p + 1)
             if (i, j) not in ((0, 0), (p, p))]          # skip n (public) and y*y (unused)
    n = len(df)
    space = (dp.vector_domain(dp.atom_domain(T=float, nan=False), size=n), dp.symmetric_distance())
    A, b, epsilon_spent = np.zeros((p, p)), np.zeros(p), 0.0
    A[0, 0] = n
    for i, j in pairs:
        corners = [u * v for u in bounds[i] for v in bounds[j]]
        lo, hi = float(min(corners)), float(max(corners))

        def make(s, lo=lo, hi=hi):
            return space >> dp.t.then_clamp(bounds=(lo, hi)) >> dp.t.then_sum() >> dp.m.then_laplace(scale=s)

        meas = make(dp.binary_search_param(make, d_in=2, d_out=epsilon / len(pairs)))
        epsilon_spent += meas.map(2)                    # one changed row touches every entry
        value = meas((cols[i] * cols[j]).tolist())
        if j == p:
            b[i] = value
        else:
            A[i, j] = A[j, i] = value
    assert epsilon_spent <= epsilon + 1e-6
    return np.linalg.lstsq(A, b, rcond=None)[0], epsilon_spent   # solve = post-processing


def _test():
    import pandas as pd
    x = np.linspace(-5, 5, 400)
    df = pd.DataFrame({"x1": x, "x2": x[::-1] / 2, "y": 3 + 2 * x - x[::-1] / 2})
    coef, eps = dp_ols_multi(df, ["x1", "x2"], "y", [(-5.0, 5.0), (-2.5, 2.5)], (-10.0, 30.0), epsilon=1.0)
    assert coef.shape == (3,)
    return {"epsilon": float(eps)}
