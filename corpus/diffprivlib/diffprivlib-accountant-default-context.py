import warnings

import numpy as np
from diffprivlib import models, tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_mean_then_classifier(df, value_column, bounds, feature_columns, label_column, bounds_X, classes, epsilon,
                            shares=(0.5, 0.5)):
    """A DP mean and a DP GaussianNB fit, both charged to one accountant made active by a `with` block.

    Inside the block, diffprivlib calls without accountant= spend on it; the previous default is restored on exit.
    (Without the block they would spend on a global default with an INFINITE budget.) Replace-one (n public).
    bounds, bounds_X=(lower_vec, upper_vec) and classes are public. Returns ((mean, model), epsilon_spent).
    """
    eps_mean, eps_model = (s * epsilon for s in shares)
    lo, hi = (np.broadcast_to(np.asarray(b, float), (len(feature_columns),)) for b in bounds_X)
    X = np.clip(df[list(feature_columns)].to_numpy(dtype=float), lo, hi) - lo   # public shift: lower bound 0
    y = df[label_column].to_numpy()
    # GaussianNB noises only the classes present in y: one public placeholder row per class keeps all present.
    X, y = np.vstack([X, np.tile((hi - lo) / 2, (len(classes), 1))]), np.concatenate([y, np.asarray(classes)])
    with BudgetAccountant(epsilon=eps_mean + eps_model / 2, delta=0) as acc:   # capped at what it records
        private_mean = tools.mean(df[value_column].to_numpy(dtype=float), epsilon=eps_mean, bounds=bounds)
        # GaussianNB is calibrated per class (add/remove); a replaced row can switch class: pass half.
        clf = models.GaussianNB(epsilon=min(eps_model / 2, acc.remaining()[0]), bounds=(np.zeros_like(hi), hi - lo))
        clf.fit(X, y)
    clf.theta_ += lo  # post-processing: predict() takes raw features
    epsilon_spent = acc.total()[0] + eps_model / 2   # GaussianNB counted twice
    assert epsilon_spent <= epsilon + 1e-9
    return (float(private_mean), clf), epsilon_spent


def _test():
    import pandas as pd
    a = np.linspace(-5, 5, 200)
    df = pd.DataFrame({"value": a, "a": a, "b": 5 - a, "label": (a > 0).astype(int)})
    (m, clf), epsilon_spent = dp_mean_then_classifier(df, "value", (-5, 5), ["a", "b"], "label",
                                                      ([-5, 0], [5, 10]), [0, 1, 2], epsilon=1.0)
    assert -5 <= m <= 5 and list(clf.classes_) == [0, 1, 2]
    return {"epsilon": epsilon_spent}
