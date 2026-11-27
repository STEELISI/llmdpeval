import warnings

from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing range must fail, never silently read the data


def dp_joint_histogram(df, column_x, column_y, bins_x, bins_y, range_x, range_y, epsilon):
    """DP 2-D histogram of two numeric columns, replace-one (bounded DP, n public). bins and ranges=(lower, upper) per axis are public.

    Returns ((private_counts, edges_x, edges_y), epsilon_spent). Row-normalising the table is post-processing.
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    x = df[column_x].to_numpy(dtype=float)
    y = df[column_y].to_numpy(dtype=float)
    # diffprivlib calibrates histograms for add/remove (sensitivity 1 per bin). Under replace-one a changed row
    # leaves one bin and enters another, so pass epsilon/2; the guarantee is 2x what the accountant recorded.
    counts, edges_x, edges_y = tools.histogram2d(x, y, epsilon=epsilon / 2, bins=(bins_x, bins_y),
                                                 range=(range_x, range_y), accountant=acc)  # returns 3 values
    return (counts, edges_x, edges_y), 2 * acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    a = np.arange(400)
    df = pd.DataFrame({"x": a % 60, "code": a % 4})
    (counts, ex, ey), epsilon_spent = dp_joint_histogram(df, "x", "code", 6, 4, (0, 60), (0, 4), epsilon=1.0)
    assert counts.shape == (6, 4) and len(ex) == 7 and len(ey) == 5
    return {"epsilon": epsilon_spent}
