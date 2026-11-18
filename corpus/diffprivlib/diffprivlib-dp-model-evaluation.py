import warnings

import numpy as np
from diffprivlib import models, tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_train_and_evaluate(df_train, df_test, feature_columns, label_column, lower_bounds, upper_bounds, classes,
                          epsilon, model_share=0.8):
    """Train a DP GaussianNB and release its test accuracy with a DP count of correct predictions.

    Replace-one (n public). Bounds and classes are public; len(df_test) is treated as public.
    Returns ((model, private_accuracy), epsilon_spent).
    """
    lo, hi = np.asarray(lower_bounds, float), np.asarray(upper_bounds, float)
    prep = lambda d: np.clip(d[list(feature_columns)].to_numpy(dtype=float), lo, hi) - lo   # public shift
    y = df_train[label_column].to_numpy()
    # diffprivlib 0.6.6 noises only the classes present in y: one public placeholder row per class keeps all present.
    X = np.vstack([prep(df_train), np.tile((hi - lo) / 2, (len(classes), 1))])
    y = np.concatenate([y, np.asarray(classes, dtype=y.dtype)])
    eps_model, eps_eval = model_share * epsilon, (1 - model_share) * epsilon
    acc = BudgetAccountant(epsilon=eps_model / 2 + eps_eval, delta=0)   # capped at what it will record
    # GaussianNB is calibrated per class (add/remove); a replaced row can switch class: pass half.
    clf = models.GaussianNB(epsilon=eps_model / 2, accountant=acc, bounds=(np.zeros_like(hi), hi - lo)).fit(X, y)
    correct = clf.predict(prep(df_test)) == df_test[label_column].to_numpy()   # full-length test array
    dp_correct = tools.count_nonzero(correct, epsilon=min(eps_eval, acc.remaining()[0]), accountant=acc)
    private_accuracy = float(np.clip(dp_correct / len(df_test), 0, 1))  # post-processing
    epsilon_spent = acc.total()[0] + eps_model / 2   # GaussianNB counted twice
    assert epsilon_spent <= epsilon + 1e-9
    return (clf, private_accuracy), epsilon_spent


def _test():
    import pandas as pd
    a = np.linspace(-5, 5, 800)
    df = pd.DataFrame({"f1": a, "f2": 5 - a, "label": (a > 0).astype(int)})
    (clf, accuracy), epsilon_spent = dp_train_and_evaluate(df[::2], df[1::2], ["f1", "f2"], "label",
                                                           [-5, 0], [5, 10], [0, 1, 2], epsilon=1.0)
    assert 0 <= accuracy <= 1 and list(clf.classes_) == [0, 1, 2]
    return {"epsilon": epsilon_spent}
