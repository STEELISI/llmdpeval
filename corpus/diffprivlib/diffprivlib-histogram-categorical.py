import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_category_counts(df, column, categories, epsilon):
    """DP count per category, replace-one (bounded DP, n public). categories is the public list (codebook, task text) in a fixed order,
    never the column's unique values or value_counts order (that order itself leaks).

    Returns ({category: private_count}, epsilon_spent). Ranking the counts afterwards is post-processing.
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    index = {c: i for i, c in enumerate(categories)}
    codes = df[column].map(index).fillna(-1).to_numpy(dtype=float)  # values outside the list fall outside every bin
    k = len(categories)
    edges = np.arange(k + 1) - 0.5  # one bin per code 0 .. k-1
    # diffprivlib calibrates histograms for add/remove (sensitivity 1 per bin). Under replace-one a changed row
    # leaves one bin and enters another, so pass epsilon/2; the guarantee is 2x what the accountant recorded.
    counts, _ = tools.histogram(codes, epsilon=epsilon / 2, bins=edges, range=(-0.5, k - 0.5), accountant=acc)
    return dict(zip(categories, counts.tolist())), 2 * acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a", "d"] * 20})
    counts, epsilon_spent = dp_category_counts(df, "group", ["a", "b", "c", "d"], epsilon=1.0)
    assert list(counts) == ["a", "b", "c", "d"]
    return {"epsilon": epsilon_spent}
