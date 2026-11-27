import numpy as np


def dp_partition_ratios(df, partition_column, partitions, num_column, den_column, num_bounds, den_bounds,
                        epsilon, ratio_range=(0.0, 1.0), min_den=1.0, max_rel_se=0.5):
    """Per-partition sums of two bounded columns and their ratio, for many disjoint partitions (areas,
    units, sites) from a PUBLIC partition list. Replace-one neighbours (bounded DP, n public).

    num_bounds, den_bounds: PUBLIC (low, high); negative lows are allowed. ratio_range, min_den, max_rel_se: public rules.
    Returns ({"num", "den", "ratio", "noisy_ok"} arrays in partition order, epsilon_spent).
    """
    rng = np.random.default_rng()
    idx = {p: i for i, p in enumerate(partitions)}
    k = df[partition_column].map(idx)
    keep = k.notna().to_numpy()
    k = k[keep].astype(int).to_numpy()
    num = df[num_column].astype(float).clip(*num_bounds).to_numpy()[keep]
    den = df[den_column].astype(float).clip(*den_bounds).to_numpy()[keep]
    sums = np.concatenate([np.bincount(k, num, len(partitions)), np.bincount(k, den, len(partitions))])
    # One row adds (num, den) to ONE partition: L1 <= max|num| + max|den| (abs: lows may be negative). Partitions are disjoint, so all
    # of them form one vector release (parallel composition). Replace-one can move the row to another
    # partition (remove + add): twice that.
    sens = 2.0 * (max(map(abs, num_bounds)) + max(map(abs, den_bounds)))
    scale = sens / epsilon
    noisy = sums + rng.laplace(0.0, scale, sums.size)
    n_num, n_den = noisy[:len(partitions)], noisy[len(partitions):]
    # Post-processing: ratio with a public floor; flag partitions where noise dominates (delta method).
    d = np.maximum(n_den, min_den)
    ratio = np.clip(n_num / d, *ratio_range)       # valid range of the ratio, e.g. a share in [0, 1]
    rel_se = np.sqrt(2) * scale * np.sqrt(1 + ratio ** 2) / d / np.maximum(np.abs(ratio), 1e-12)
    return {"num": n_num, "den": n_den, "ratio": ratio, "noisy_ok": rel_se <= max_rel_se}, sens / scale


def _test():
    rng = np.random.default_rng()
    import pandas as pd
    m = 5000
    part = rng.choice([f"p{i}" for i in range(60)], m)
    df = pd.DataFrame({"part": part, "w": rng.random(m), "flag": rng.random(m) < 0.6})
    df["wf"] = df["w"] * df["flag"]
    res, eps = dp_partition_ratios(df, "part", [f"p{i}" for i in range(60)], "wf", "w", (0, 1), (0, 1), 1.0)
    assert res["ratio"].shape == (60,)
    return {"epsilon": float(eps)}
