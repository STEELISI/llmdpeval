import math

import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


def cap_items(cell, items, max_items, sep):
    """Distinct PUBLIC items of one unit, at most max_items chosen at random (looks only at this unit)."""
    own = list(dict.fromkeys(t for t in str(cell).split(sep) if t in items))
    return list(np.random.permutation(own)[:max_items]) if len(own) > max_items else own


def dp_two_group_item_tables(df, group_column, groups, items_column, items, max_items, n_total, epsilon,
                             group_share=0.05, sep=";"):
    """Per PUBLIC item, how many units of group a / group b list it (a 2x2 table per item). One row = one unit,
    items in one delimited column; groups (a, b), n_total (= len(df)) and max_items are PUBLIC.
    Replace-one unit (bounded DP). Returns (noisy counts + noise variances, epsilon_spent)."""
    items, keep = [str(t) for t in items], set(map(str, items))
    is_b = (df[group_column] == groups[1]).astype(int).tolist()
    tokens = [f"{g}|{t}" for g, cell in zip(is_b, df[items_column].fillna(""))
              for t in cap_items(cell, keep, max_items, sep)]
    cells = [f"{g}|{t}" for g in (0, 1) for t in items]          # public (group, item) cells
    eps_g, eps_h = group_share * epsilon, (1 - group_share) * epsilon
    size_space = dp.vector_domain(dp.atom_domain(bounds=(0, 1)), size=n_total), dp.symmetric_distance()
    make_g = lambda s: size_space >> dp.t.then_sum() >> dp.m.then_laplace(s)   # size of group b, d_in = 2
    s_g = dp.binary_search_param(make_g, d_in=2, d_out=eps_g)          # noise scale kept for the test
    d_h = 2 * max_items       # a capped unit = at most max_items tokens; replace-one = remove + add one unit
    hist = (dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()) \
        >> dp.t.then_count_by_categories(categories=cells, null_category=False)
    make_h = lambda s: hist >> dp.m.then_laplace(s)
    s_h = dp.binary_search_param(make_h, d_in=d_h, d_out=eps_h)
    m_g, m_h = make_g(s_g), make_h(s_h)
    n_b = float(np.clip(m_g(is_b), 1, n_total - 1))
    noisy = np.array(m_h(tokens), dtype=float).reshape(2, len(items))
    eps_spent = m_g.map(2) + m_h.map(d_h)                     # sequential composition, framework accounting
    return {"A_a": noisy[0], "A_b": noisy[1], "n_a": n_total - n_b, "n_b": n_b,
            "var_cell": 2 * s_h ** 2, "var_n": 2 * s_g ** 2}, eps_spent   # discrete Laplace var ~ 2 scale^2


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
