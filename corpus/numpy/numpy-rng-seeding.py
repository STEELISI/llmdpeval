import numpy as np


def _laplace_count(rng, x, value, epsilon):
    """One Laplace count (Δ = 1 under add/remove and replace-one) from the generator passed in."""
    scale = 1.0 / epsilon
    return float(np.sum(x == value) + rng.laplace(0.0, scale)), 1.0 / scale


def dp_counts_where(df, column, values, epsilon, rng=None):
    """DP counts for several public values, all noise from ONE unseeded generator.

    Returns ({value: private_count}, epsilon_spent). Budget split evenly over the values.
    """
    if rng is None:
        rng = np.random.default_rng()    # no argument: fresh OS entropy, unpredictable noise
    # Create the generator once and pass it down. Never re-create a generator per call
    # (repeated noise) and never use the legacy global np.random functions.
    x = df[column].to_numpy()
    eps_each = epsilon / len(values)
    results, epsilon_spent = {}, 0.0
    for v in values:
        results[v], spent = _laplace_count(rng, x, v, eps_each)
        epsilon_spent += spent           # sequential composition
    assert epsilon_spent <= epsilon + 1e-9
    return results, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    r1, epsilon_spent = dp_counts_where(df, "group", ["a", "b"], epsilon=1.0)
    r2, _ = dp_counts_where(df, "group", ["a", "b"], epsilon=1.0)
    assert r1 != r2                      # unseeded: two releases differ
    return {"epsilon": epsilon_spent}
