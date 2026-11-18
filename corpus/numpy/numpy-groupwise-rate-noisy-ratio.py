import numpy as np


def dp_group_rates(df, group_column, groups, flag_column, flag_value, epsilon,
                   relation="replace_one", num_share=0.5, min_den=1.0):
    """DP rate of flag_value within each group of a PUBLIC group list: noisy count / noisy count.

    Returns ({group: {"rate", "noisy_size"}}, epsilon_spent). min_den is a PUBLIC floor.
    relation: "replace_one" (bounded DP, n public; default) or "add_remove". Group sizes are private.
    """
    rng = np.random.default_rng()
    g = df[group_column].to_numpy()
    hit = df[flag_column].to_numpy() == flag_value
    num = np.array([np.sum(hit[g == c]) for c in groups], dtype=float)   # zeros included
    den = np.array([np.sum(g == c) for c in groups], dtype=float)
    # Each vector is a histogram: one record is in one group. L1 sensitivity 1 under
    # add/remove, 2 under replace-one (the record can change group or flag).
    l1 = 1.0 if relation == "add_remove" else 2.0
    scale_num = l1 / (num_share * epsilon)
    scale_den = l1 / ((1 - num_share) * epsilon)
    noisy_num = num + rng.laplace(0.0, scale_num, size=len(groups))
    noisy_den = den + rng.laplace(0.0, scale_den, size=len(groups))
    rate = np.clip(np.maximum(noisy_num, 0) / np.maximum(noisy_den, min_den), 0, 1)  # post-processing
    epsilon_spent = l1 / scale_num + l1 / scale_den        # sequential composition
    assert epsilon_spent <= epsilon + 1e-9
    out = {c: {"rate": float(r), "noisy_size": float(d)} for c, r, d in zip(groups, rate, noisy_den)}
    return out, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"g": ["a", "b", "a", "c"] * 50, "flag": ["y", "n", "n", "y"] * 50})
    res, eps = dp_group_rates(df, "g", ["a", "b", "c", "d"], "flag", "y", epsilon=1.0)
    assert all(0 <= v["rate"] <= 1 for v in res.values())
    return {"epsilon": eps}
