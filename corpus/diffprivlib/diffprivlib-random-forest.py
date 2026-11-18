import warnings

from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # missing bounds or classes must fail, never read the data


def dp_random_forest(df, feature_columns, label_column, bounds, classes, epsilon, n_estimators=10, max_depth=5):
    """DP random forest classifier. bounds=(lower_vec, upper_vec) are public per-feature bounds (data is clipped);
    classes is the public label list from the task. Rows are split disjointly among the trees, so more trees
    means fewer rows each; memory grows exponentially with max_depth. Returns (fitted model, epsilon_spent).
    """
    acc = BudgetAccountant()
    X = df[list(feature_columns)].to_numpy(dtype=float)
    y = df[label_column].to_numpy()
    # Replace-one: a changed row can move between two leaves, each a separate release at the passed epsilon.
    rf = models.RandomForestClassifier(n_estimators=n_estimators, epsilon=epsilon / 2, bounds=bounds, classes=classes,
                                       max_depth=max_depth, shuffle=True, accountant=acc)
    rf.fit(X, y)  # predict() and score() afterwards are post-processing
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return rf, epsilon_spent


def _test():
    import numpy as np
    import pandas as pd
    a = np.linspace(0, 10, 400)
    df = pd.DataFrame({"a": a, "b": 10 - a, "label": (a > 5).astype(int)})
    rf, epsilon_spent = dp_random_forest(df, ["a", "b"], "label", bounds=([0, 0], [10, 10]), classes=[0, 1],
                                         epsilon=1.0, n_estimators=3, max_depth=3)
    assert rf.predict(df[["a", "b"]].to_numpy()).shape == (400,)
    return {"epsilon": epsilon_spent}
