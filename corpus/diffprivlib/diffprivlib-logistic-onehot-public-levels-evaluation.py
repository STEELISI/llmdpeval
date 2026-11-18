import warnings

import numpy as np
from diffprivlib import models, tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing data_norm must fail, never read the data


def design_matrix(df, numeric, categorical):
    """numeric {col: (low, high)} PUBLIC bounds -> [0, 1]; categorical {col: [PUBLIC levels]} -> one-hot
    (unknown values -> all zeros). Same columns for train and test whatever they contain. Each block adds
    <= 1 to the squared norm, so scaling by 1/sqrt(2d) plus an intercept column 1/sqrt(2) gives norm <= 1."""
    blocks = [((df[c].astype(float).clip(lo, hi) - lo) / (hi - lo)).to_numpy()[:, None] for c, (lo, hi) in numeric.items()]
    blocks += [np.column_stack([(df[c] == lv).to_numpy(float) for lv in levels]) for c, levels in categorical.items()]
    d = len(numeric) + len(categorical)
    return np.column_stack([b / np.sqrt(2 * d) for b in blocks] + [np.full(len(df), 1 / np.sqrt(2))])


def dp_logistic_with_group_accuracy(train, test, label, numeric, categorical, group_column, groups,
                                    epsilon, model_share=0.8, C=1.0):
    """DP logistic regression (objective perturbation) on numeric + one-hot features, then overall and
    per-group test accuracy from ONE noisy histogram of (group, correct) cells. Replace-one (bounded DP:
    train and test sizes public, group sizes private). Returns ({"model", "accuracy", "group_accuracy"}, eps).
    """
    acc = BudgetAccountant()
    C = min(C, 4 * np.expm1(model_share * epsilon / 8))   # public rule: slack term <= a quarter of the share
    clf = models.LogisticRegression(epsilon=model_share * epsilon, data_norm=1.0, C=C, fit_intercept=False,
                                    max_iter=5000, accountant=acc)
    clf.fit(design_matrix(train, numeric, categorical), train[label].to_numpy())
    correct = clf.predict(design_matrix(test, numeric, categorical)) == test[label].to_numpy()
    g = test[group_column].map({v: i for i, v in enumerate(groups)}).fillna(len(groups)).to_numpy()
    cells = 2 * g + correct                      # cell = (group incl. "other", correct?)
    k = 2 * (len(groups) + 1)
    # A changed test row leaves one cell and enters another; diffprivlib charges 1 per row, so pass half.
    eps_eval = (1 - model_share) * epsilon
    noisy, _ = tools.histogram(cells, epsilon=eps_eval / 2, bins=k, range=(-0.5, k - 0.5), accountant=acc)
    noisy = np.maximum(noisy.astype(float), 0.0).reshape(-1, 2)   # rows: group; columns: wrong, right
    accuracy = float(np.clip(noisy[:, 1].sum() / len(test), 0, 1))  # test size is public
    group_acc = {v: float(np.clip(noisy[i, 1] / max(noisy[i].sum(), 1.0), 0, 1)) for i, v in enumerate(groups)}
    epsilon_spent = model_share * epsilon + eps_eval     # = accountant total + eps_eval/2 (replace-one)
    assert abs(epsilon_spent - (acc.total()[0] + eps_eval / 2)) < 1e-9
    return {"model": clf, "accuracy": accuracy, "group_accuracy": group_acc}, epsilon_spent


def _test():
    import pandas as pd
    m = 2000
    df = pd.DataFrame({"h": np.random.uniform(0, 80, m), "k": np.random.choice(["s", "t", "u"], m),
                       "grp": np.random.choice(["g1", "g2"], m)})
    df["y"] = ((df["h"] > 40) ^ (np.random.rand(m) < 0.1)).astype(int)
    res, eps = dp_logistic_with_group_accuracy(df[::2], df[1::2], "y", {"h": (0, 100)}, {"k": ["s", "t", "u"]},
                                               "grp", ["g1", "g2"], epsilon=1.0)
    assert 0 <= res["accuracy"] <= 1 and set(res["group_accuracy"]) == {"g1", "g2"}
    return {"epsilon": float(eps)}
