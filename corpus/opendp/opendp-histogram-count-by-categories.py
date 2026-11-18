import opendp.prelude as dp

dp.enable_features("contrib")


def dp_histogram(df, column, categories, epsilon, null_category=True):
    """One noisy count per public category, replace-one neighbours (bounded DP, n public). The whole
    histogram is one release: a changed row leaves one cell and enters another (L1 sensitivity 2).

    Returns ({category: count}, epsilon_spent); "other" holds values not in categories.
    """
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    hist = space >> dp.t.then_count_by_categories(categories=categories, null_category=null_category)
    meas = dp.binary_search_chain(lambda s: hist >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)   # replace-one = symmetric distance 2 (one removal + one addition)
    noisy = meas(df[column].astype(str).tolist())
    labels = list(categories) + (["other"] if null_category else [])
    return dict(zip(labels, noisy)), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "z"] * 20})
    counts, epsilon_spent = dp_histogram(df, "group", ["a", "b", "c"], epsilon=1.0)
    assert list(counts) == ["a", "b", "c", "other"]
    return {"epsilon": float(epsilon_spent)}
