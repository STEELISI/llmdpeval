import warnings

import numpy as np
from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_scaler_pca_logistic(df, feature_columns, label_column, bounds, n_components, data_norm_pca,
                           data_norm_clf, epsilon, shares=(1 / 3, 1 / 3, 1 / 3)):
    """sklearn Pipeline where every step that learns from data is its diffprivlib version, on one accountant.
    Replace-one (n public): scaler (public n), PCA and objective perturbation need no halving.

    bounds=(lower_vec, upper_vec): public per-feature bounds for the scaler. data_norm_pca / data_norm_clf: public
    L2-norm bounds on the rows each step sees. shares (sum 1) split epsilon across the three steps.
    Returns (fitted pipeline, epsilon_spent). Every refit spends again.
    Before the classifier, a fixed public map scales rows by 1/(sqrt(2)*data_norm_clf) and appends an intercept
    column 1/sqrt(2), so the classifier sees norm <= 1 with fit_intercept=False (0.6.6 under-charges epsilon
    for row norms above 1, and fit_intercept=True raises the norm to sqrt(q^2 + 1)).
    """
    def unit_rows(Z):  # public, data-independent: no fit, no budget
        return np.column_stack([Z / (np.sqrt(2) * data_norm_clf), np.full(len(Z), 1 / np.sqrt(2))])

    eps_scale, eps_pca, eps_clf = (s * epsilon for s in shares)
    C = min(1.0, 4 * np.expm1(eps_clf / 8))   # public rule: slack 2*ln(1 + C/4) uses <= eps_clf/4
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    pipe = Pipeline([
        ("scaler", models.StandardScaler(epsilon=eps_scale, bounds=bounds, accountant=acc)),
        ("pca", models.PCA(n_components=n_components, epsilon=eps_pca, data_norm=data_norm_pca, centered=True,
                           accountant=acc)),
        ("unit", FunctionTransformer(unit_rows)),
        ("clf", models.LogisticRegression(epsilon=eps_clf, data_norm=1.0, C=C, fit_intercept=False,
                                          accountant=acc)),
    ])
    pipe.fit(df[list(feature_columns)].to_numpy(dtype=float), df[label_column].to_numpy())
    return pipe, acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    a = np.linspace(0, 10, 300)
    df = pd.DataFrame({"a": a, "b": 10 - a, "c": a / 2, "label": (a > 5).astype(int)})
    pipe, epsilon_spent = dp_scaler_pca_logistic(df, ["a", "b", "c"], "label", ([0, 0, 0], [10, 10, 5]), 2,
                                                 data_norm_pca=5.0, data_norm_clf=5.0, epsilon=1.0)
    assert pipe.predict(df[["a", "b", "c"]].to_numpy()).shape == (300,)
    return {"epsilon": epsilon_spent}
