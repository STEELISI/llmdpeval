import warnings

import numpy as np
from diffprivlib import models, tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_bounds_then_naive_bayes(df, feature_columns, label_column, classes, edges, threshold, epsilon,
                               bounds_share=0.5):
    """When no public bounds exist: estimate per-feature bounds with DP histograms, then fit GaussianNB.

    Replace-one (n public). edges: public, data-independent bin edges (e.g. a log scale covering any plausible
    magnitude). threshold: public noise threshold a noisy bin count must exceed. classes: public label list.
    Both steps are add/remove-calibrated (histogram bins, per-class statistics), so each gets half its share
    and the accountant total is reported twice. Returns (((lowers, uppers), model), epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon / 2, delta=0)   # records the halved shares
    eps_bounds, eps_model = bounds_share * epsilon, (1 - bounds_share) * epsilon
    X = df[list(feature_columns)].to_numpy(dtype=float)
    lowers, uppers = [], []
    for j in range(X.shape[1]):
        counts, e = tools.histogram(X[:, j], epsilon=min(eps_bounds / X.shape[1] / 2, acc.remaining()[0]),
                                    bins=edges, range=(edges[0], edges[-1]), accountant=acc)
        hit = np.flatnonzero(counts > threshold)
        lowers.append(e[hit[0]] if hit.size else e[0])  # nothing above threshold: fall back to the public edges
        uppers.append(e[hit[-1] + 1] if hit.size else e[-1])
    lo, hi = np.asarray(lowers, float), np.asarray(uppers, float)
    X = np.clip(X, lo, hi) - lo                       # shift by the DP lower bounds: class sums need lower bound 0
    y = df[label_column].to_numpy()
    # diffprivlib 0.6.6 noises only the classes present in y: one placeholder row per public class keeps all present.
    X, y = np.vstack([X, np.tile((hi - lo) / 2, (len(classes), 1))]), np.concatenate([y, np.asarray(classes)])
    clf = models.GaussianNB(epsilon=min(eps_model / 2, acc.remaining()[0]), bounds=(np.zeros_like(hi), hi - lo),
                            accountant=acc)
    clf.fit(X, y)
    clf.theta_ += lo                                   # post-processing: predict() takes raw features
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return ((lowers, uppers), clf), epsilon_spent


def _test():
    import pandas as pd
    a = np.linspace(-30, 30, 600)
    df = pd.DataFrame({"a": a, "b": 100 - a, "label": (a > 0).astype(int)})
    edges = [-(2.0 ** p) for p in range(10, -1, -1)] + [0.0] + [2.0 ** p for p in range(11)]
    (bnds, clf), epsilon_spent = dp_bounds_then_naive_bayes(df, ["a", "b"], "label", [0, 1], edges, threshold=10,
                                                            epsilon=1.0)
    assert all(lo < hi for lo, hi in zip(*bnds)) and clf.predict(df[["a", "b"]].to_numpy()).shape == (600,)
    return {"epsilon": epsilon_spent}
