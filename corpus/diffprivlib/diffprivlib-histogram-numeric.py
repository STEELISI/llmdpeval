import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_histogram(df, column, bins, bounds, epsilon):
    """DP histogram of a numeric column, replace-one (bounded DP, n public). bins (an int or an explicit edge list) and bounds=(lower, upper)
    are fixed from public knowledge before looking at the data.

    Returns ((private_counts, bin_edges), epsilon_spent). Normalising the counts later is post-processing.
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    values = df[column].to_numpy(dtype=float)
    # diffprivlib calibrates histograms for add/remove (sensitivity 1 per bin). Under replace-one a changed row
    # leaves one bin and enters another, so pass epsilon/2; the guarantee is 2x what the accountant recorded.
    counts, edges = tools.histogram(values, epsilon=epsilon / 2, bins=bins, range=bounds, accountant=acc)
    return (counts, edges), 2 * acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(500) % 90})
    (counts, edges), epsilon_spent = dp_histogram(df, "value", bins=9, bounds=(0, 90), epsilon=1.0)
    assert len(counts) == 9 and len(edges) == 10
    return {"epsilon": epsilon_spent}
