import warnings

import numpy as np
from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_linear_regression(df, feature_columns, target_column, bounds_X, bounds_y, epsilon):
    """DP linear regression of df[target_column] on df[feature_columns]. Replace-one (n public).

    bounds_X=(lower, upper): scalars, or one value per feature. bounds_y=(lower, upper) for the target.
    Both must be public (codebook, physical limits), never the data's min/max; data is clipped to them.
    diffprivlib 0.6.6 behaviour handled here: (a) fit_intercept=True centres X and y with extra DP means that the
    accountant does not record; (b) the X'X diagonal sensitivity is taken from the lower bound only. So: no
    library intercept, features shifted to [-(U-L), 0] (lower bound has the largest magnitude), and an explicit
    constant column -1 carries the intercept. Returns ({"coef", "intercept", "model"}, epsilon_spent);
    predictions are X_new @ coef + intercept (post-processing).
    """
    d = len(feature_columns)
    lo, hi = (np.broadcast_to(np.asarray(b, float), (d,)) for b in bounds_X)
    X = np.clip(df[list(feature_columns)].to_numpy(dtype=float), lo, hi) - hi       # in [-(U-L), 0]
    X = np.column_stack([X, -np.ones(len(X))])                                       # intercept column
    y = df[target_column].to_numpy(dtype=float)
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    reg = models.LinearRegression(epsilon=epsilon, bounds_X=(np.append(lo - hi, -1.0), np.zeros(d + 1)),
                                  bounds_y=bounds_y, fit_intercept=False, accountant=acc)
    reg.fit(X, y)
    coef = reg.coef_[:d]
    intercept = float(-reg.coef_[d] - coef @ hi)                                     # undo the public shift
    epsilon_spent = acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return {"coef": coef, "intercept": intercept, "model": reg}, epsilon_spent


def _test():
    import pandas as pd
    x = np.linspace(-5, 5, 2000)
    df = pd.DataFrame({"x": x, "y": 2 * x + 1})
    res, epsilon_spent = dp_linear_regression(df, ["x"], "y", bounds_X=(-5, 5), bounds_y=(-10, 12), epsilon=1.0)
    assert res["coef"].shape == (1,) and np.isfinite(res["intercept"])
    assert abs(res["model"]._obj_coefs[2][0, 0] - ((x - 5) ** 2).sum()) > 1e-6   # the X'X diagonal is noised
    return {"epsilon": epsilon_spent}
