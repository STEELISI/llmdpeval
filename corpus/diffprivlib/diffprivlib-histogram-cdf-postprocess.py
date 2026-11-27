import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing range must fail, never silently read the data


def dp_cdf_and_quantiles(df, column, bounds, n_bins, quantiles, epsilon):
    """DP CDF on a public grid, replace-one (bounded DP, n public). bounds=(lower, upper) are public.

    Only the histogram uses data; everything after it is post-processing, so any number of quantiles is free.
    Returns ((edges, cdf, {q: value}), epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    # diffprivlib calibrates for add/remove; replace-one moves a row between two bins: pass epsilon/2, report 2x.
    counts, edges = tools.histogram(df[column].to_numpy(dtype=float), epsilon=epsilon / 2, bins=n_bins,
                                    range=bounds, accountant=acc)
    total = max(counts.sum(), 1)  # noisy total, not the true n
    cdf = np.cumsum(counts) / total
    qs = {q: float(edges[min(np.searchsorted(cdf, q) + 1, len(edges) - 1)]) for q in quantiles}
    return (edges, cdf, qs), 2 * acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(1000) % 100})
    (edges, cdf, qs), epsilon_spent = dp_cdf_and_quantiles(df, "value", (0, 100), 20, [0.1, 0.5, 0.9], epsilon=1.0)
    assert np.all(np.diff(cdf) >= 0) and len(qs) == 3
    return {"epsilon": epsilon_spent}
