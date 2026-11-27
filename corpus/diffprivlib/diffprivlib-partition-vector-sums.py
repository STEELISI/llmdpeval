import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Laplace


def row_vectors(df, value_columns, row_l1_bound):
    """Per-row contribution vector (non-negative columns, e.g. one-hot group flags, flag x outcome, membership
    probabilities). Missing values count as 0 (a public rule). Rows whose L1 norm exceeds the PUBLIC bound are
    scaled down to it, so one row adds at most row_l1_bound to the released vector."""
    v = df[list(value_columns)].to_numpy(dtype=float, copy=True)     # copy: pandas 3 arrays can be read-only
    v = np.clip(np.nan_to_num(v, nan=0.0), 0.0, None)
    l1 = v.sum(axis=1, keepdims=True)
    return v * np.minimum(1.0, row_l1_bound / np.maximum(l1, 1e-12))


def dp_partition_vector_sums(df, partition_column, partitions, value_columns, row_l1_bound, epsilon):
    """Sums of several bounded columns in every partition of a PUBLIC list (thousands of disjoint areas, sites,
    parts), released together as one vector. Replace-one row (bounded DP, n public).
    Returns (noisy sums: array partitions x columns, laplace scale, epsilon_spent)."""
    idx = {p: i for i, p in enumerate(partitions)}
    k = df[partition_column].map(idx)
    keep = k.notna().to_numpy()                                      # rows outside the public list are dropped
    v = row_vectors(df.loc[keep], value_columns, row_l1_bound)
    sums = np.zeros((len(partitions), len(value_columns)))
    np.add.at(sums, k[keep].astype(int).to_numpy(), v)
    # One row sits in ONE partition and adds at most C = row_l1_bound in L1. Replace-one removes it and adds
    # another row, maybe in another partition: L1 sensitivity 2C for the whole partitions x columns vector.
    # Laplace with that sensitivity on every entry is ONE release: record one spend (tools.* would loop).
    acc = BudgetAccountant()
    mech = Laplace(epsilon=epsilon, sensitivity=2.0 * row_l1_bound)
    noisy = np.vectorize(mech.randomise)(sums)
    acc.spend(epsilon, 0)
    return noisy, np.sqrt(mech.variance(0.0) / 2), acc.total()[0]


def ratio_of_columns(noisy, num_col, den_col, floor=1.0, valid=(0.0, 1.0)):
    """Post-processing: per-partition ratio of two released columns with a public floor and valid range."""
    return np.clip(noisy[:, num_col] / np.maximum(noisy[:, den_col], floor), *valid)


def ecological_regression(noisy, total_col, outcome_col, group_col, region, n_regions, floor=5.0):
    """Post-processing: per region, regress each partition's outcome rate on its group share (both from noisy
    sums; partitions with a noisy total below a public floor are skipped). Rate of the group = a + b, of the
    rest = a. Noisy shares bias the slope toward 0 (errors in variables)."""
    tot = noisy[:, total_col]
    ok = tot >= floor
    y = np.clip(noisy[:, outcome_col] / np.maximum(tot, floor), 0, 1)
    x = np.clip(noisy[:, group_col] / np.maximum(tot, floor), 0, 1)
    out = np.full((n_regions, 2), np.nan)
    for r in range(n_regions):
        s = ok & (np.asarray(region) == r)
        if s.sum() >= 3 and np.ptp(x[s]) > 0:
            b, a = np.polyfit(x[s], y[s], 1)
            out[r] = np.clip([a + b, a], 0, 1)
    return out


def _test():
    import pandas as pd
    m, parts = 20000, [f"p{i}" for i in range(300)]
    g = np.random.choice(3, m, p=[0.6, 0.3, 0.1])
    df = pd.DataFrame({"part": np.random.choice(parts, m), "out": (np.random.rand(m) < 0.6).astype(np.int8)})
    for j in range(3):
        df[f"g{j}"] = (g == j).astype(np.int8)                      # one-hot group flags
        df[f"g{j}_out"] = df[f"g{j}"] * df["out"]                   # flag x outcome
    soft = np.random.dirichlet(np.ones(3), m)
    soft[:5] = np.nan                                               # a few missing values
    for j in range(3):
        df[f"s{j}"], df[f"s{j}_out"] = soft[:, j], soft[:, j] * df["out"]
    cols = [f"g{j}" for j in range(3)] + [f"g{j}_out" for j in range(3)] + \
        [f"s{j}" for j in range(3)] + [f"s{j}_out" for j in range(3)]
    noisy, scale, eps = dp_partition_vector_sums(df, "part", parts, cols, row_l1_bound=4.0, epsilon=1.0)
    assert noisy.shape == (300, 12) and row_vectors(df, cols, 4.0).sum(axis=1).max() <= 4.0 + 1e-9
    rate = ratio_of_columns(noisy, 3, 0)
    region = np.zeros((10, 12))
    np.add.at(region, np.arange(300) % 10, noisy)                 # coarser regions = sums of partitions
    assert rate.shape == (300,)
    eco = ecological_regression(noisy, 0, 3, 0, np.arange(300) % 10, 10)
    assert eco.shape == (10, 2)
    return {"epsilon": float(eps)}
