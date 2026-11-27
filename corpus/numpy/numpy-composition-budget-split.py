import numpy as np


def dp_sums_with_budget(df, columns, bounds, epsilon, shares=None, relation="replace_one"):
    """Several DP sums (one per column) under ONE total epsilon, basic sequential composition.

    bounds: {column: (lower, upper)}, public. shares: per-query fractions of epsilon (default equal).
    Returns ({column: private_sum}, epsilon_spent).
    """
    rng = np.random.default_rng()
    shares = shares or [1.0 / len(columns)] * len(columns)
    eps_list = [s * epsilon for s in shares]                 # fix the split before touching the data
    assert sum(eps_list) <= epsilon + 1e-9, "split overspends the budget"
    results, ledger = {}, 0.0
    for col, eps_i in zip(columns, eps_list):
        lower, upper = bounds[col]
        x = np.clip(df[col].to_numpy(dtype=float), lower, upper)
        # sum sensitivity: replace-one (default, n public) U - L; add/remove (n private) max(|L|, |U|)
        sens = upper - lower if relation == "replace_one" else max(abs(lower), abs(upper))
        scale = sens / eps_i
        spent = sens / scale
        if ledger + spent > epsilon + 1e-9:                  # ledger: later edits cannot overspend
            raise RuntimeError("privacy budget exceeded")
        ledger += spent
        results[col] = float(np.sum(x) + rng.laplace(0.0, scale))
    return results, ledger                                   # total = sum of per-query epsilons


def _test():
    import pandas as pd
    df = pd.DataFrame({"a": [1.0, 4.0, 2.5, 9.0] * 25, "b": [-3.0, 0.5, 2.0, 1.0] * 25})
    res, epsilon_spent = dp_sums_with_budget(df, ["a", "b"], {"a": (0.0, 10.0), "b": (-5.0, 5.0)},
                                             epsilon=1.0, shares=[0.7, 0.3])
    assert set(res) == {"a", "b"}
    wide = pd.DataFrame({f"c{i}": [-4.0, 1.0, 3.0] * 10 for i in range(40)})   # k = 40 equal shares
    res40, eps40 = dp_sums_with_budget(wide, list(wide), {c: (-5.0, 5.0) for c in wide}, epsilon=1.0)
    assert len(res40) == 40 and eps40 <= 1.0 + 1e-9
    return {"epsilon": max(epsilon_spent, eps40)}
