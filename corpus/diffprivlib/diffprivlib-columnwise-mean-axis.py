import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_column_means(df, columns, lower_bounds, upper_bounds, epsilon):
    """DP mean of each column in `columns`. lower_bounds/upper_bounds: one public value per column.

    axis=0 makes diffprivlib split epsilon evenly across the columns internally.
    Returns (dict column -> private mean, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    X = df[list(columns)].to_numpy(dtype=float)
    bounds = (np.asarray(lower_bounds, dtype=float), np.asarray(upper_bounds, dtype=float))  # tuple of 2 vectors
    means = tools.mean(X, epsilon=epsilon, bounds=bounds, axis=0, accountant=acc)
    return dict(zip(columns, map(float, means))), acc.total()[0]


def _test():
    import pandas as pd
    a = np.arange(300) % 50
    df = pd.DataFrame({"a": a, "b": a * 2.0, "c": 100 - a})
    means, epsilon_spent = dp_column_means(df, ["a", "b", "c"], [0, 0, 0], [50, 100, 100], epsilon=1.0)
    assert len(means) == 3 and 0 <= means["a"] <= 50
    return {"epsilon": epsilon_spent}
