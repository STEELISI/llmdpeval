import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import histogram


def dp_top_k_by_rate(df, column, candidates, k, epsilon, outcome_column, outcome_value):
    """Top-k public candidates by the RATE of rows with outcome_value, pure DP, replace-one (bounded DP, n public).
    candidates: PUBLIC list of every possible value, declared before looking at the data (codebook,
    schema), in a fixed order; never df[column].unique() (which values occur is itself private).

    Two diffprivlib histograms over the public candidate list (as integer codes 0..K-1), epsilon/2
    each (passed as epsilon/4): outcome rows per group and all rows per group. The ratio is post-processing.
    Returns ({"top": [...], "noisy_rates": {...}}, epsilon_spent).
    """
    acc = BudgetAccountant()
    index = {c: i for i, c in enumerate(candidates)}
    codes = df[column].map(index)                       # groups outside the public list -> NaN
    hit = (df[outcome_column] == outcome_value)
    keep_rows = codes.notna()
    K = len(candidates)
    # public range and one bin per code: no data-dependent bins (no PrivacyLeakWarning)
    # diffprivlib calibrates for add/remove (sensitivity 1 per bin); replace-one moves a row between two bins,
    # so pass half of each share. The guarantee is 2x what the accountant recorded.
    opts = dict(epsilon=epsilon / 4, bins=K, range=(-0.5, K - 0.5), accountant=acc)
    noisy_num, _ = histogram(codes[keep_rows & hit].to_numpy(), **opts)
    noisy_den, _ = histogram(codes[keep_rows].to_numpy(), **opts)
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    # Post-processing: drop groups whose NOISY size is below 10 noise scales (scale 1/(epsilon/4)).
    scale = 1.0 / (epsilon / 4)
    noisy_num, noisy_den = noisy_num.astype(float), noisy_den.astype(float)
    rate = np.where(noisy_den >= 10 * scale, noisy_num / np.maximum(noisy_den, 1.0), -np.inf)
    order = [i for i in np.argsort(-rate) if np.isfinite(rate[i])][:k]   # groups below the cut-off are never picked
    return {"top": [candidates[i] for i in order],
            "noisy_rates": dict(zip(candidates, rate.tolist()))}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 300 + ["b"] * 300 + ["c"] * 300 + ["d"] * 3,
                       "flag": (["y"] * 240 + ["n"] * 60) + (["y"] * 90 + ["n"] * 210)
                               + (["y"] * 150 + ["n"] * 150) + ["y"] * 3})
    res, eps = dp_top_k_by_rate(df, "group", ["a", "b", "c", "d"], k=2, epsilon=1.0,
                                outcome_column="flag", outcome_value="y")
    assert len(res["top"]) == 2
    return {"epsilon": float(eps)}
