import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Laplace


def dp_partition_ratios(df, partition_column, partitions, num_column, den_column, num_bounds, den_bounds,
                        epsilon, ratio_range=(0.0, 1.0), min_den=1.0, max_rel_se=0.5):
    """Per-partition sums of two bounded columns and their ratio, for many disjoint partitions (areas,
    units, sites) from a PUBLIC partition list. Replace-one neighbours (bounded DP, n public).

    num_bounds, den_bounds: PUBLIC (low, high), low >= 0. ratio_range, min_den, max_rel_se: public rules.
    Returns ({"num", "den", "ratio", "noisy_ok"} arrays in partition order, epsilon_spent).
    """
    assert num_bounds[0] >= 0 and den_bounds[0] >= 0, "sensitivity below assumes lower bounds >= 0"
    idx = {p: i for i, p in enumerate(partitions)}
    k = df[partition_column].map(idx)
    keep = k.notna().to_numpy()
    k = k[keep].astype(int).to_numpy()
    num = df[num_column].astype(float).clip(*num_bounds).to_numpy()[keep]
    den = df[den_column].astype(float).clip(*den_bounds).to_numpy()[keep]
    sums = np.concatenate([np.bincount(k, num, len(partitions)), np.bincount(k, den, len(partitions))])
    # One row adds (num, den) to ONE partition: L1 <= max(num) + max(den). Partitions are disjoint, so all
    # of them form one vector release (parallel composition). Replace-one can move the row to another
    # partition (remove + add): twice that.
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)
    mech = Laplace(epsilon=epsilon, sensitivity=2.0 * (num_bounds[1] + den_bounds[1]))
    noisy = np.array([mech.randomise(float(v)) for v in sums])   # L1 vector release: this noise per entry
    acc.spend(epsilon, 0)                          # one release for all partitions
    scale = np.sqrt(mech.variance(0.0) / 2)        # Laplace scale, from the mechanism
    n_num, n_den = noisy[:len(partitions)], noisy[len(partitions):]
    # Post-processing: ratio with a public floor; flag partitions where noise dominates (delta method).
    d = np.maximum(n_den, min_den)
    ratio = np.clip(n_num / d, *ratio_range)       # valid range of the ratio, e.g. a share in [0, 1]
    rel_se = np.sqrt(2) * scale * np.sqrt(1 + ratio ** 2) / d / np.maximum(np.abs(ratio), 1e-12)
    epsilon_spent = acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return {"num": n_num, "den": n_den, "ratio": ratio, "noisy_ok": rel_se <= max_rel_se}, epsilon_spent


def _test():
    import pandas as pd
    m = 5000
    part = np.random.choice([f"p{i}" for i in range(60)], m)
    df = pd.DataFrame({"part": part, "w": np.random.rand(m), "flag": np.random.rand(m) < 0.6})
    df["wf"] = df["w"] * df["flag"]
    res, eps = dp_partition_ratios(df, "part", [f"p{i}" for i in range(60)], "wf", "w", (0, 1), (0, 1), 1.0)
    assert res["ratio"].shape == (60,)
    return {"epsilon": float(eps)}
