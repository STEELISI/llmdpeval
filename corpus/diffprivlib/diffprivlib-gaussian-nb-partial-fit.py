import warnings

import numpy as np
from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_incremental_naive_bayes(batches, feature_columns, label_column, lower_bounds, upper_bounds, classes, epsilon):
    """GaussianNB updated batch by batch with partial_fit. Bounds and classes are public. Replace-one (n public).

    epsilon is the total: each batch gets epsilon / len(batches), and the accountant records every batch
    (conservative: correct even if a person appears in several batches). Returns (model, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon / 2, delta=0)  # the accountant sees the halved shares
    lo, hi = np.asarray(lower_bounds, float), np.asarray(upper_bounds, float)
    # Replace-one: a changed row can switch class (two classes' statistics change): pass half.
    share = epsilon / len(batches) / 2
    clf = models.GaussianNB(epsilon=share, accountant=acc, bounds=(np.zeros_like(hi), hi - lo))
    pad_X, pad_y = np.tile((hi - lo) / 2, (len(classes), 1)), np.asarray(classes)
    for i, batch in enumerate(batches):
        X = np.clip(batch[list(feature_columns)].to_numpy(dtype=float), lo, hi) - lo  # public shift
        y = batch[label_column].to_numpy()
        # diffprivlib 0.6.6 noises only the classes present in the batch; an empty class would show as count 0.
        # One public placeholder row per class keeps every class present in every batch.
        X, y = np.vstack([X, pad_X]), np.concatenate([y, pad_y.astype(y.dtype)])
        clf.epsilon = min(share, acc.remaining()[0])  # B equal shares can round above the cap
        clf.partial_fit(X, y, classes=list(classes) if i == 0 else None)  # classes required on the first call
    clf.theta_ += lo  # post-processing: predict() takes raw features
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return clf, epsilon_spent


def _test():
    import pandas as pd
    a = np.linspace(-5, 5, 600)
    df = pd.DataFrame({"f1": a, "f2": 5 - a, "label": (a > 0).astype(int)})
    batches = [df[i::3] for i in range(3)] + [df[df["label"] == 0]]   # last batch has no row of class 1
    clf, epsilon_spent = dp_incremental_naive_bayes(batches, ["f1", "f2"], "label", [-5, 0], [5, 10], [0, 1], 1.0)
    assert len(clf.predict(df[["f1", "f2"]].to_numpy()[:5])) == 5
    return {"epsilon": epsilon_spent}
