import warnings

import numpy as np
from diffprivlib import models, tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_clusters_with_sizes(df, columns, lower_bounds, upper_bounds, n_clusters, epsilon, fit_share=0.7):
    """DP KMeans centres plus DP cluster sizes, one accountant. Bounds and n_clusters are public.

    Assigning rows to clusters reads private data again, so the sizes are a second DP release.
    Replace-one (bounded DP, n public): a changed row can move between two clusters.
    Returns ((centres, sizes), epsilon_spent).
    """
    acc = BudgetAccountant()
    X = df[list(columns)].to_numpy(dtype=float)
    lo, hi = np.asarray(lower_bounds, float), np.asarray(upper_bounds, float)
    X = np.clip(X, lo, hi) - lo   # public shift: lower bound 0 (per-cluster sum sensitivity is U - L)
    # KMeans releases per-cluster counts and sums for add/remove; a moved row changes two clusters: pass half.
    km = models.KMeans(n_clusters=n_clusters, epsilon=fit_share * epsilon / 2, accountant=acc,
                       bounds=(np.zeros_like(hi), hi - lo))
    km.fit(X)
    labels = km.predict(X)  # private: never release these or their exact counts
    eps_sizes = (1 - fit_share) * epsilon
    # The histogram is calibrated for add/remove (1 bin); a moved row changes 2 bins: pass half, add the other half.
    sizes, _ = tools.histogram(labels, epsilon=eps_sizes / 2, bins=n_clusters,
                               range=(0, n_clusters), accountant=acc)
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return (km.cluster_centers_ + lo, sizes), epsilon_spent


def _test():
    import pandas as pd
    a = np.arange(600)
    df = pd.DataFrame({"x": (a % 3) * 4 + (a % 7) / 7, "y": (a % 3) * 4 + (a % 5) / 5})
    (centres, sizes), epsilon_spent = dp_clusters_with_sizes(df, ["x", "y"], [0, 0], [10, 10], 3, epsilon=1.0)
    assert centres.shape == (3, 2) and len(sizes) == 3
    return {"epsilon": epsilon_spent}
