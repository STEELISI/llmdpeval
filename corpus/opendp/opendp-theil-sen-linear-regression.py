import numpy as np
import opendp.prelude as dp
from opendp.extras.sklearn.linear_model import LinearRegression

# The model is built from user transformations, so "honest-but-curious" is required.
dp.enable_features("contrib", "honest-but-curious")


def dp_linear_fit(df, x_column, y_column, x_bounds, y_bounds, x_new, epsilon):
    """DP Theil-Sen fit y ~ slope * x + intercept for ONE feature, then predict at x_new.

    x_bounds=(LX, UX) and y_bounds=(LY, UY) must be public. Replace-one (n public): d_in = 2
    (one changed row = symmetric distance 2; add/remove-one, n private: d_in = 1, map(1)).
    Returns ({"slope", "intercept", "predictions"}, epsilon_spent).
    """
    def make(scale):                                  # the constructor takes scale, not epsilon
        return LinearRegression(dp.max_divergence(), x_bounds=[x_bounds], y_bounds=y_bounds,
                                scale=scale).measurement

    scale = dp.binary_search_param(make, d_in=2, d_out=epsilon)
    model = LinearRegression(dp.max_divergence(), x_bounds=[x_bounds], y_bounds=y_bounds, scale=scale)
    epsilon_spent = model.measurement.map(2)
    assert epsilon_spent <= epsilon
    X = df[[x_column]].to_numpy(dtype=float)          # shape (n, 1)
    fitted = model.fit(X, df[y_column].to_numpy(dtype=float))   # a plain sklearn model
    predictions = fitted.predict(np.asarray(x_new, dtype=float).reshape(-1, 1))  # post-processing
    result = {"slope": float(fitted.coef_[0]), "intercept": float(fitted.intercept_),
              "predictions": predictions.tolist()}
    return result, epsilon_spent


def _test():
    import pandas as pd
    x = np.linspace(0, 10, 200)
    df = pd.DataFrame({"x": x, "y": 2 * x + 1 + np.random.normal(0, 1, 200)})
    _, epsilon_spent = dp_linear_fit(df, "x", "y", (0.0, 10.0), (-5.0, 30.0), [1.0, 5.0], 1.0)
    return {"epsilon": float(epsilon_spent)}
