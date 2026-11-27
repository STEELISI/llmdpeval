import numpy as np


def dp_group_means(df, group_column, groups, value_column, lower, upper, epsilon,
                   relation="replace_one", sum_share=0.5, min_count=1.0):
    """DP mean of a column clamped to PUBLIC [lower, upper] for each group of a PUBLIC list.

    Disjoint groups -> parallel composition: each vector costs its epsilon once.
    relation: "replace_one" (bounded DP, n public; default) or "add_remove". Group sizes are private.
    Returns ({group: mean}, epsilon_spent).
    """
    rng = np.random.default_rng()
    g = df[group_column].to_numpy()
    v = np.clip(df[value_column].to_numpy(dtype=float), lower, upper)
    s = np.array([v[g == c].sum() for c in groups])
    n = np.array([np.sum(g == c) for c in groups], dtype=float)
    # add/remove: one record touches one group. replace-one: it may move groups, so double.
    factor = 1.0 if relation == "add_remove" else 2.0
    sens_sum = factor * max(abs(lower), abs(upper))
    sens_cnt = factor * 1.0
    scale_sum = sens_sum / (sum_share * epsilon)
    scale_cnt = sens_cnt / ((1 - sum_share) * epsilon)
    noisy_s = s + rng.laplace(0.0, scale_sum, size=len(groups))
    noisy_n = n + rng.laplace(0.0, scale_cnt, size=len(groups))
    means = np.clip(noisy_s / np.maximum(noisy_n, min_count), lower, upper)   # post-processing
    epsilon_spent = sens_sum / scale_sum + sens_cnt / scale_cnt
    assert epsilon_spent <= epsilon + 1e-9
    return dict(zip(groups, means.tolist())), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"g": ["a", "b", "a", "c"] * 50, "v": [3.0, 8.0, 5.0, 1.0] * 50})
    res, eps = dp_group_means(df, "g", ["a", "b", "c"], "v", 0.0, 10.0, epsilon=1.0)
    assert all(0 <= m <= 10 for m in res.values())
    return {"epsilon": eps}
