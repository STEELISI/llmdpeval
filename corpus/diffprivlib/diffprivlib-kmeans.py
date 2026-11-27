import warnings

import numpy as np
from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_kmeans(df, feature_columns, n_clusters, bounds, epsilon):
    """DP k-means centres of df[feature_columns].

    bounds=(lower_vec, upper_vec): public per-feature bounds; data is clipped to them and the initial centres are
    drawn inside them. Put features on comparable public scales first. Returns (cluster_centers_, epsilon_spent).
    """
    acc = BudgetAccountant()
    lo, hi = (np.broadcast_to(np.asarray(b, float), (len(feature_columns),)) for b in bounds)
    X = np.clip(df[list(feature_columns)].to_numpy(dtype=float), lo, hi) - lo   # public shift: lower bound 0
    # Replace-one: a changed row can move between two clusters (two counts and sums change): pass half.
    km = models.KMeans(n_clusters=n_clusters, epsilon=epsilon / 2, bounds=(np.zeros_like(hi), hi - lo), accountant=acc)
    km.fit(X)  # km.predict(X_new - lo) afterwards is post-processing
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return km.cluster_centers_ + lo, epsilon_spent


def _test():
    import pandas as pd
    a = np.concatenate([np.full(100, 1.0), np.full(100, 9.0)])
    df = pd.DataFrame({"a": a, "b": a})
    centres, epsilon_spent = dp_kmeans(df, ["a", "b"], n_clusters=2, bounds=([0, 0], [10, 10]), epsilon=1.0)
    assert centres.shape == (2, 2)
    return {"epsilon": epsilon_spent}
