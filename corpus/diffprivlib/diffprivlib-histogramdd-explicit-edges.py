import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # data-derived edges must fail, never silently leak


def dp_data_cube(df, columns, edges, epsilon):
    """DP d-dimensional histogram, replace-one (bounded DP, n public). edges: one public, sorted edge array per column (no range needed).

    Returns ((private_cube, edge_list), epsilon_spent). Marginals via cube.sum(axis=k) are post-processing.
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    sample = df[list(columns)].to_numpy(dtype=float)
    # diffprivlib calibrates histograms for add/remove (sensitivity 1 per bin). Under replace-one a changed row
    # leaves one bin and enters another, so pass epsilon/2; the guarantee is 2x what the accountant recorded.
    cube, edge_list = tools.histogramdd(sample, epsilon=epsilon / 2, bins=[np.asarray(e, float) for e in edges],
                                        accountant=acc)
    return (cube, edge_list), 2 * acc.total()[0]


def _test():
    import pandas as pd
    a = np.arange(500)
    df = pd.DataFrame({"x": a % 30, "y": a % 7, "z": a % 3})
    edges = [[0, 10, 20, 30], [0, 2, 4, 7], [0, 1, 2, 3]]
    (cube, e), epsilon_spent = dp_data_cube(df, ["x", "y", "z"], edges, epsilon=1.0)
    assert cube.shape == (3, 3, 3) and cube.sum(axis=2).shape == (3, 3)
    return {"epsilon": epsilon_spent}
