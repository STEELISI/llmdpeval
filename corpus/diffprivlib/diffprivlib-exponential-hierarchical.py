import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_perturb_hierarchical(df, column, hierarchy, epsilon):
    """Local-DP perturbation of a leaf category using a public nested-list hierarchy of strings.

    Leaves in the same branch are likelier substitutes. Returns (perturbed values, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = mechanisms.ExponentialHierarchical(epsilon=epsilon, hierarchy=hierarchy)
    noisy = np.array([mech.randomise(str(v)) for v in df[column]])
    acc.spend(epsilon, 0)  # one randomise call per person
    return noisy, acc.total()[0]


def _test():
    import pandas as pd
    hierarchy = [["a1", "a2"], ["b1", "b2"]]
    df = pd.DataFrame({"area": [["a1", "a2", "b1", "b2"][i % 4] for i in range(200)]})
    noisy, epsilon_spent = dp_perturb_hierarchical(df, "area", hierarchy, epsilon=1.0)
    assert set(noisy) <= {"a1", "a2", "b1", "b2"}
    return {"epsilon": epsilon_spent}
