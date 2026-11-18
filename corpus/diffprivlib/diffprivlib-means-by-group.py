import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import count_nonzero, sum as dp_sum


def dp_means_by_group(df, group_column, groups, value_column, bounds, epsilon, min_size=1.0):
    """DP mean of value_column in each PUBLIC group, replace-one (bounded DP, n public): noisy sum / noisy count.

    groups: public list declared before looking at the data; bounds: PUBLIC (L, U).
    Every call gets a FULL-LENGTH array (length n, public) with non-members set to 0: diffprivlib truncates
    sum/count_nonzero outputs to a range set by the array length, so a per-group subset would expose the
    private group size. Returns ({group: mean}, epsilon_spent).
    """
    lo, hi = bounds
    v = df[value_column].to_numpy(float).clip(lo, hi) - lo          # in [0, U-L]
    g_col = df[group_column].to_numpy()
    out, spent = {}, []
    for g in groups:
        in_g = g_col == g
        acc = BudgetAccountant(epsilon=epsilon / 2, delta=0)        # one accountant per group
        noisy_sum = dp_sum(np.where(in_g, v, 0.0), epsilon=epsilon / 4, bounds=(0.0, hi - lo), accountant=acc)
        noisy_n = count_nonzero(in_g, epsilon=min(epsilon / 4, acc.remaining()[0]), accountant=acc)
        out[g] = float(np.clip(noisy_sum / max(noisy_n, min_size) + lo, lo, hi))   # post-processing
        spent.append(acc.total()[0])
    # A replaced row can leave one group and join another: two groups change, so twice the largest group cost.
    epsilon_spent = 2 * max(spent)
    assert epsilon_spent <= epsilon + 1e-9
    return out, epsilon_spent


def _test():
    import pandas as pd
    GROUPS = ["g1", "g2", "g3"]   # public list, declared up front; g3 has no rows
    df = pd.DataFrame({"grp": ["g1"] * 100 + ["g2"] * 100, "val": [-3.0, 5.0] * 50 + [1.0, 2.0] * 50})
    outs = [dp_means_by_group(df, "grp", GROUPS, "val", (-5.0, 5.0), epsilon=1.0) for _ in range(30)]
    assert all(set(m) == set(GROUPS) for m, _ in outs)
    assert len({m["g3"] for m, _ in outs}) > 1   # an empty group must not always give the same output
    return {"epsilon": float(outs[0][1])}
