import warnings

import numpy as np
from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_feature_scaling(df, columns, lower_bounds, upper_bounds, epsilon):
    """DP per-feature mean and std via diffprivlib's StandardScaler. Bounds (one per feature) are public.

    Half of epsilon goes to the means, half to the variances (split inside diffprivlib).
    Returns ((means, stds, fitted_scaler), epsilon_spent). The scaler can transform data; the transformed rows stay private.
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    scaler = models.StandardScaler(epsilon=epsilon, accountant=acc,
                                   bounds=(np.asarray(lower_bounds, float), np.asarray(upper_bounds, float)))
    scaler.fit(df[list(columns)].to_numpy(dtype=float))
    return (dict(zip(columns, scaler.mean_)), dict(zip(columns, scaler.scale_)), scaler), acc.total()[0]


def _test():
    import pandas as pd
    a = np.arange(400) % 40
    df = pd.DataFrame({"a": a, "b": a * 3.0})
    (means, stds, scaler), epsilon_spent = dp_feature_scaling(df, ["a", "b"], [0, 0], [40, 120], epsilon=1.0)
    assert set(means) == {"a", "b"} and all(s >= 0 for s in stds.values())
    return {"epsilon": epsilon_spent}
