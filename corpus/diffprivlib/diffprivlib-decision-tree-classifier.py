import warnings

import numpy as np
from diffprivlib import models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # missing bounds/classes must fail, never silently read the data


def dp_decision_tree(df_train, df_test, feature_columns, label_column, lower_bounds, upper_bounds, classes,
                     epsilon, max_depth=4):
    """Train a DP decision tree; bounds (one value per feature) and classes are public.

    Returns ((model, test_accuracy), epsilon_spent). Test accuracy here is non-private.
    """
    acc = BudgetAccountant()
    # Replace-one: a changed row can move between two leaves, each a separate release at the passed epsilon.
    tree = models.DecisionTreeClassifier(max_depth=max_depth, epsilon=epsilon / 2,
                                         bounds=(np.asarray(lower_bounds, float), np.asarray(upper_bounds, float)),
                                         classes=list(classes), accountant=acc)
    tree.fit(df_train[list(feature_columns)].to_numpy(dtype=float), df_train[label_column].to_numpy())
    pred = tree.predict(df_test[list(feature_columns)].to_numpy(dtype=float))
    accuracy = float(np.mean(pred == df_test[label_column].to_numpy()))
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return (tree, accuracy), epsilon_spent


def _test():
    import pandas as pd
    a = np.linspace(0, 10, 400)
    df = pd.DataFrame({"f1": a, "f2": 10 - a, "label": (a > 5).astype(int)})
    (tree, accuracy), epsilon_spent = dp_decision_tree(df[:300], df[300:], ["f1", "f2"], "label",
                                                       [0, 0], [10, 10], [0, 1], epsilon=1.0)
    assert 0 <= accuracy <= 1
    return {"epsilon": epsilon_spent}
