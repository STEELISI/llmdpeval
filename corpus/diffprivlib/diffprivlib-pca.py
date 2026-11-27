import warnings

from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_pca(df, feature_columns, n_components, bounds, data_norm, epsilon):
    """DP PCA of df[feature_columns]; the library centres the data with a DP mean (half of epsilon).

    bounds=(lower_vec, upper_vec): public per-feature bounds, used for that DP mean.
    data_norm: public bound on the L2 norm of a centred row, e.g. the norm of (upper_vec - lower_vec).
    Returns ((components_, explained_variance_ratio_), epsilon_spent). Component signs are arbitrary.
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    X = df[list(feature_columns)].to_numpy(dtype=float)
    pca = models.PCA(n_components=n_components, epsilon=epsilon, data_norm=data_norm, centered=False,
                     bounds=bounds, accountant=acc)
    pca.fit(X)  # pca.transform(X_new) afterwards is post-processing
    return (pca.components_, pca.explained_variance_ratio_), acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    a = np.linspace(0, 10, 300)
    df = pd.DataFrame({"a": a, "b": 0.5 * a, "c": 10 - a})
    lower, upper = np.zeros(3), np.full(3, 10.0)
    (components, ratio), epsilon_spent = dp_pca(df, ["a", "b", "c"], 2, bounds=(lower, upper),
                                                data_norm=float(np.linalg.norm(upper - lower)), epsilon=1.0)
    assert components.shape == (2, 3)
    return {"epsilon": epsilon_spent}
