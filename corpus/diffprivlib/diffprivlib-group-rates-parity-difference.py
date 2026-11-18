import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import histogram


def dp_group_rates_parity(df, group_column, groups, outcome_column, outcome_value, group_a, group_b,
                          epsilon, min_size=1.0):
    """Per-group outcome rates and the statistical parity difference rate[a] - rate[b], pure DP, replace-one (bounded DP, n public).

    groups: PUBLIC list of every group value, declared before looking at the data; never unique().
    Two histograms over integer codes 0..K-1, epsilon/2 each (passed as epsilon/4, see below). Returns ({"rates", "noisy_sizes", "spd"}, eps).
    """
    acc = BudgetAccountant()
    codes = df[group_column].map({c: i for i, c in enumerate(groups)})   # outside the list -> NaN
    keep = codes.notna()
    hit = df[outcome_column] == outcome_value
    K = len(groups)
    # diffprivlib calibrates histograms for add/remove (sensitivity 1 per bin). Under replace-one a changed row
    # leaves one bin and enters another, so pass half of each share; the guarantee is 2x what the accountant recorded.
    opts = dict(epsilon=epsilon / 4, bins=K, range=(-0.5, K - 0.5), accountant=acc)   # public range
    noisy_pos, _ = histogram(codes[keep & hit].to_numpy(), **opts)
    noisy_all, _ = histogram(codes[keep].to_numpy(), **opts)
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    noisy_pos, noisy_all = noisy_pos.astype(float), noisy_all.astype(float)
    rates = dict(zip(groups, np.clip(noisy_pos / np.maximum(noisy_all, min_size), 0.0, 1.0).tolist()))
    return {"rates": rates, "noisy_sizes": dict(zip(groups, noisy_all.tolist())),
            "spd": rates[group_a] - rates[group_b]}, epsilon_spent


def _test():
    import pandas as pd
    GROUPS = ["g1", "g2", "g3"]   # public list, declared up front
    df = pd.DataFrame({"grp": ["g1"] * 200 + ["g2"] * 200, "pred": [1, 0] * 100 + [1, 0, 0, 0] * 50})
    res, eps = dp_group_rates_parity(df, "grp", GROUPS, "pred", 1, "g1", "g2", epsilon=1.0)
    assert -1.0 <= res["spd"] <= 1.0
    return {"epsilon": float(eps)}
