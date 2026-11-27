import math

import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Laplace
from diffprivlib.tools import count_nonzero


def cap_items(cell, items, max_items, sep):
    """Distinct PUBLIC items of one unit, at most max_items chosen at random (looks only at this unit)."""
    own = list(dict.fromkeys(t for t in str(cell).split(sep) if t in items))
    return list(np.random.permutation(own)[:max_items]) if len(own) > max_items else own


def dp_two_group_item_tables(df, group_column, groups, items_column, items, max_items, n_total, epsilon,
                             group_share=0.05, sep=";"):
    """Per PUBLIC item, how many units of group a / group b list it (a 2x2 table per item). One row = one unit,
    items in one delimited column; groups (a, b), n_total and max_items are PUBLIC.
    Replace-one unit (bounded DP). Returns (noisy counts + noise variances, epsilon_spent)."""
    items, col = list(items), {t: j for j, t in enumerate(items)}
    is_b = (df[group_column] == groups[1]).to_numpy()
    counts = np.zeros((2, len(items)))
    for g, cell in zip(is_b.astype(int), df[items_column].fillna("")):
        for t in cap_items(cell, col, max_items, sep):
            counts[g, col[t]] += 1
    acc = BudgetAccountant()
    eps_g, eps_h = group_share * epsilon, (1 - group_share) * epsilon
    # Size of group b: replace-one changes it by at most 1 when n is public, so count_nonzero's
    # sensitivity 1 is right here. n_a = n_total - n_b.
    n_b = float(np.clip(count_nonzero(is_b, epsilon=eps_g, accountant=acc), 1, n_total - 1))
    # One (group, item) histogram as one vector: a capped unit adds at most max_items; replace-one removes one
    # unit and adds another (maybe in the other group): L1 sensitivity 2 * max_items. tools.histogram would
    # assume sensitivity 1, so use the Laplace mechanism on every entry and record ONE spend.
    mech = Laplace(epsilon=eps_h, sensitivity=2.0 * max_items)
    noisy = np.array([[mech.randomise(float(v)) for v in row] for row in counts])
    acc.spend(eps_h, 0)
    return {"A_a": noisy[0], "A_b": noisy[1], "n_a": n_total - n_b, "n_b": n_b,
            "var_cell": mech.variance(0.0), "var_n": 2 / eps_g ** 2}, acc.total()[0]


def noise_aware_tests(r, alpha=0.05, min_total_scales=5.0):
    """Post-processing only. Two-proportion score (Pearson) test per item with the DP noise variance added;
    items tested = noisy total >= a public multiple of the noise scale; Bonferroni over the tested items."""
    A_a, A_b, n_a, n_b = np.maximum(r["A_a"], 0), np.maximum(r["A_b"], 0), r["n_a"], r["n_b"]
    n, pi, T = n_a + n_b, n_b / (n_a + n_b), A_a + A_b
    D = A_b - pi * T                                          # observed minus expected under "no difference"
    var_sampling = T * pi * (1 - pi) * (n - T) / (n - 1)
    var_noise = ((1 - pi) ** 2 + pi ** 2) * r["var_cell"] + (T / n) ** 2 * r["var_n"]
    z = D / np.sqrt(np.maximum(var_sampling + var_noise, 1e-12))
    p = np.array([math.erfc(abs(v) / math.sqrt(2)) for v in z])
    tested = T >= min_total_scales * math.sqrt(r["var_cell"] / 2)
    significant = tested & (p <= alpha / max(tested.sum(), 1))
    ratio = ((A_b + 0.5) * (n_a - A_a + 0.5)) / ((n_b - A_b + 0.5) * (A_a + 0.5))
    return {"p": p, "tested": tested, "significant": significant, "ratio": ratio}


def _test():
    import pandas as pd
    items = [f"i{j}" for j in range(30)]
    n = 6000
    grp = np.random.choice(["a", "b"], n)
    rows = []
    for g in grp:
        p = np.full(30, 0.05)
        p[:3] = 0.25 if g == "b" else 0.05                   # items i0-i2 differ between groups
        rows.append(";".join(t for t, hit in zip(items, np.random.rand(30) < p) if hit))
    df = pd.DataFrame({"grp": grp, "lst": rows})
    r, eps = dp_two_group_item_tables(df, "grp", ("a", "b"), "lst", items, max_items=4, n_total=n, epsilon=1.0)
    res = noise_aware_tests(r)
    assert len(res["p"]) == 30 and res["significant"][:3].all()
    return {"epsilon": float(eps)}
