import warnings

import numpy as np
from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing data_norm must fail, never read the data


def _design_matrix(df, feature_columns, bounds):
    """Clamp each column to its PUBLIC (low, high), min-max to [0, 1], scale by 1/sqrt(2d), and add a
    constant intercept column 1/sqrt(2). Any row then has L2 norm <= 1 by construction."""
    d = len(feature_columns)
    cols = [(df[c].astype(float).clip(lo, hi) - lo) / (hi - lo) / np.sqrt(2 * d)
            for c, (lo, hi) in zip(feature_columns, bounds)]
    return np.column_stack(cols + [np.full(len(df), 1 / np.sqrt(2))])


def dp_logistic_regression_unit_norm(df, feature_columns, bounds, label_column, epsilon, C=1.0):
    """Binary DP logistic regression (objective perturbation, CMS 2011 Alg. 2) on public-bounded features.

    bounds: one PUBLIC (low, high) per feature column (documentation, codebook), never data min/max.
    The intercept is an explicit, penalised column, so fit_intercept=False (an unpenalised intercept
    leaves the objective not strongly convex in that direction). C is capped by a public rule so the
    slack term 2*ln(1 + C/4) uses at most epsilon/4. Returns ({"model", "C"}, epsilon_spent).
    """
    C = min(C, 4 * np.expm1(epsilon / 8))      # public rule, fixed before any fit; never tuned on results
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    clf = models.LogisticRegression(epsilon=epsilon, data_norm=1.0, C=C, fit_intercept=False,
                                    max_iter=5000, accountant=acc)
    clf.fit(_design_matrix(df, feature_columns, bounds), df[label_column].to_numpy())
    # Report it next to sklearn LogisticRegression(C=C, fit_intercept=False) on the same matrix (the
    # control): the gap to the control is the DP cost; the gap control -> default C is regularisation.
    return {"model": clf, "C": C}, acc.total()[0]


def _test():
    import pandas as pd
    a = np.linspace(0, 50, 400)
    df = pd.DataFrame({"a": a, "b": a[::-1] * 2, "label": (a > 25).astype(int)})
    res, epsilon_spent = dp_logistic_regression_unit_norm(df, ["a", "b"], [(0, 50), (0, 100)], "label", 1.0)
    X = _design_matrix(df, ["a", "b"], [(0, 50), (0, 100)])
    assert np.linalg.norm(X, axis=1).max() <= 1 + 1e-12 and res["C"] <= 4 * np.expm1(1 / 8)
    assert 0 <= res["model"].score(X, df["label"]) <= 1
    return {"epsilon": epsilon_spent}
