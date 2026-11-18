import warnings

import numpy as np
from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning
from diffprivlib.validation import clip_to_bounds, clip_to_norm

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing data_norm must fail, never silently read the data


def _rescale(X, lower, upper):
    """Public rescaling to [0, 1]^d, divide by sqrt(2d), append an intercept column 1/sqrt(2): row norm <= 1."""
    X01 = (clip_to_bounds(X, (lower, upper)) - lower) / (upper - lower)
    X01 = np.column_stack([X01 / np.sqrt(2 * X.shape[1]), np.full(len(X), 1 / np.sqrt(2))])
    return clip_to_norm(X01, 1.0)


def dp_multiclass_logreg(df_train, df_test, feature_columns, label_column, lower_bounds, upper_bounds, epsilon,
                         C=1.0):
    """One-vs-rest DP logistic regression; data_norm = 1 follows from the public bounds, not from the data.
    Replace-one (n public): objective perturbation with row norm <= 1 needs no halving.
    Explicit intercept column + fit_intercept=False: diffprivlib 0.6.6 charges its slack with the row norm q
    where the proof needs q^2, exact only for q <= 1 (fit_intercept=True would make q = sqrt(2)).

    Returns ((model, test_accuracy), epsilon_spent). Test accuracy here is non-private.
    """
    lower, upper = np.asarray(lower_bounds, float), np.asarray(upper_bounds, float)
    n_classes = len(np.unique(df_train[label_column]))   # the label set is assumed public
    # Public rule: each one-vs-rest model gets epsilon/K; cap C so its slack 2*ln(1 + C/4) uses <= a quarter of it.
    C = min(C, 4 * np.expm1(epsilon / max(n_classes, 2) / 8))
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    clf = models.LogisticRegression(epsilon=epsilon, data_norm=1.0, C=C, fit_intercept=False,
                                    accountant=acc)
    clf.fit(_rescale(df_train[list(feature_columns)].to_numpy(dtype=float), lower, upper),
            df_train[label_column].to_numpy())
    pred = clf.predict(_rescale(df_test[list(feature_columns)].to_numpy(dtype=float), lower, upper))
    return (clf, float(np.mean(pred == df_test[label_column].to_numpy()))), acc.total()[0]


def _test():
    import pandas as pd
    a = np.linspace(0, 9, 900)
    df = pd.DataFrame({"f1": a, "f2": 9 - a, "label": (a // 3).astype(int)})
    (clf, accuracy), epsilon_spent = dp_multiclass_logreg(df[::2], df[1::2], ["f1", "f2"], "label",
                                                          [0, 0], [9, 9], epsilon=1.0)
    assert clf.coef_.shape == (3, 3)  # 2 features + intercept column
    return {"epsilon": epsilon_spent}
