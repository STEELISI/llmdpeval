import warnings

import numpy as np
from diffprivlib import models, tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # missing bounds must fail, never read the data


def dp_repeated_split_accuracy(df, feature_columns, label_column, lower_bounds, upper_bounds, classes, repeats,
                               epsilon, test_share=0.3, model_share=0.8):
    """R train/test repeats under ONE budget: rows go to R disjoint chunks by a random permutation of row
    positions (independent of the values); each chunk is split into train/test, trains a DP GaussianNB and
    releases a DP test accuracy. Replace-one, n public: a changed row sits in one chunk only, so the total is the
    largest chunk cost (parallel composition by position). Returns ((accuracies, mean, std), epsilon_spent)."""
    lo, hi = np.asarray(lower_bounds, float), np.asarray(upper_bounds, float)
    X = np.clip(df[list(feature_columns)].to_numpy(dtype=float), lo, hi) - lo   # public shift: bounds [0, hi-lo]
    y = df[label_column].to_numpy()
    # diffprivlib 0.6.6 noises only the classes present in y: one public placeholder row per class keeps all present.
    pad_X, pad_y = np.tile((hi - lo) / 2, (len(classes), 1)), np.asarray(classes, dtype=y.dtype)
    eps_model, eps_eval = model_share * epsilon, (1 - model_share) * epsilon
    accs, costs = [], []
    for chunk in np.array_split(np.random.permutation(len(df)), repeats):
        acc = BudgetAccountant(epsilon=eps_model / 2 + eps_eval, delta=0)   # capped at what it records
        n_test = int(len(chunk) * test_share)                 # public: depends on n and the share only
        te, tr = chunk[:n_test], chunk[n_test:]
        # GaussianNB is calibrated per class (add/remove); a replaced row can switch class: pass half.
        clf = models.GaussianNB(epsilon=eps_model / 2, bounds=(np.zeros_like(hi), hi - lo),
                                accountant=acc).fit(np.vstack([X[tr], pad_X]), np.concatenate([y[tr], pad_y]))
        correct = clf.predict(X[te]) == y[te]                  # one test row moves this count by <= 1
        dp_correct = tools.count_nonzero(correct, epsilon=min(eps_eval, acc.remaining()[0]), accountant=acc)
        accs.append(float(np.clip(dp_correct / n_test, 0, 1)))
        costs.append(acc.total()[0] + eps_model / 2)   # NB counted twice under replace-one
    epsilon_spent = max(costs)                                 # chunks are disjoint by position
    assert epsilon_spent <= epsilon + 1e-9
    return (accs, float(np.mean(accs)), float(np.std(accs))), epsilon_spent


def _test():
    import pandas as pd
    rng = np.random.default_rng()
    a = rng.uniform(20, 60, 6000)
    df = pd.DataFrame({"f1": a, "f2": rng.uniform(1, 9, 6000), "label": (a > 40).astype(int)})
    (accs, mean, std), eps = dp_repeated_split_accuracy(df, ["f1", "f2"], "label", [20, 1], [60, 9], [0, 1], 5, 1.0)
    assert len(accs) == 5 and 0.6 < mean <= 1
    return {"epsilon": eps}
