import warnings

import numpy as np
from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_naive_bayes(df, feature_columns, label_column, bounds, classes, epsilon, priors=None):
    """DP Gaussian naive Bayes classifier on numeric df[feature_columns]. Replace-one (n public).

    bounds=(lower_vec, upper_vec): public per-feature bounds (or scalars); features are clipped to them.
    classes: the public list of class labels. priors: public class priors only if the task gives them.
    Returns (fitted model, epsilon_spent).
    """
    lo, hi = (np.broadcast_to(np.asarray(b, float), (len(feature_columns),)) for b in bounds)
    X = np.clip(df[list(feature_columns)].to_numpy(dtype=float), lo, hi) - lo   # public shift: lower bound 0
    y = df[label_column].to_numpy()
    if not np.isin(y, classes).all():
        raise ValueError("label outside the public class list")
    # diffprivlib 0.6.6 noises only the classes present in y; an empty class would show as count 0.
    # One public placeholder row per class (at the public midpoint) keeps every class present.
    X = np.vstack([X, np.tile((hi - lo) / 2, (len(classes), 1))])
    y = np.concatenate([y, np.asarray(classes, dtype=y.dtype)])
    acc = BudgetAccountant(epsilon=epsilon / 2, delta=0)   # capped at what it will record
    # Replace-one: a changed row can switch class and change two classes' statistics: pass half.
    clf = models.GaussianNB(epsilon=epsilon / 2, bounds=(np.zeros_like(hi), hi - lo), priors=priors, accountant=acc)
    clf.fit(X, y)
    clf.theta_ += lo  # post-processing: predict() and score() now take raw features
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return clf, epsilon_spent


def _test():
    import pandas as pd
    a = np.linspace(-5, 5, 200)
    df = pd.DataFrame({"a": a, "b": 5 - a, "label": (a > 0).astype(int)})
    # class 2 is public but has no rows: it must still get a noisy count, never an exact 0
    clf, epsilon_spent = dp_naive_bayes(df, ["a", "b"], "label", ([-5, 0], [5, 10]), [0, 1, 2], epsilon=1.0)
    assert list(clf.classes_) == [0, 1, 2] and clf.class_count_.min() >= 1
    assert clf.predict(df[["a", "b"]].to_numpy()).shape == (200,)
    return {"epsilon": epsilon_spent}
