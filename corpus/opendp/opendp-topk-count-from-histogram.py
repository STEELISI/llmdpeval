import opendp.prelude as dp

dp.enable_features("contrib")


def dp_top_k_by_count(df, column, categories, k, epsilon, replace_one=True):
    """Top-k public categories by COUNT: one noisy histogram over the public list, then sort.

    categories: public list of every possible value, fixed before looking at the data (codebook, schema;
    never df[column].unique()), e.g. sorted alphabetically; ties in
    the noisy counts are broken by this order, so it must not come from the data.
    replace_one: True (default) for bounded DP (n public; one changed row moves two bins, d_in = 2),
    False for add/remove (d_in = 1). Returns ({"top": [...], "noisy_counts": {...}}, epsilon_spent).
    """
    d_in = 2 if replace_one else 1
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    hist = space >> dp.t.then_count_by_categories(categories=categories, null_category=False)
    # The framework finds the scale for the whole vector (L1 sensitivity), whatever k is.
    meas = dp.binary_search_chain(lambda s: hist >> dp.m.then_laplace(s), d_in=d_in, d_out=epsilon)
    epsilon_spent = meas.map(d_in)
    noisy = meas(df[column].astype(str).tolist())
    # Post-processing on released values only: sort by noisy count, ties by the public list order.
    ranked = sorted(range(len(categories)), key=lambda i: (-noisy[i], i))
    return {"top": [categories[i] for i in ranked[:k]],
            "noisy_counts": dict(zip(categories, noisy))}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 50 + ["b"] * 30 + ["c"] * 20 + ["d"] * 1 + ["zz"] * 5})
    res, eps1 = dp_top_k_by_count(df, "group", ["a", "b", "c", "d"], k=2, epsilon=1.0)
    _, eps2 = dp_top_k_by_count(df, "group", ["a", "b", "c", "d"], k=2, epsilon=1.0, replace_one=False)
    assert len(res["top"]) == 2 and set(res["top"]) <= {"a", "b", "c", "d"}
    return {"epsilon": float(max(eps1, eps2))}
