import opendp.prelude as dp

dp.enable_features("contrib")


def dp_histogram_gaussian(df, column, categories, epsilon, delta):
    """Histogram over a public category list with Gaussian noise. The count vector's sensitivity is
    measured in L2, which is smaller than L1 when one person touches many bins.
    Replace-one neighbours (bounded DP, n public): d_in = 2. Result is (epsilon, delta)-DP via zCDP.

    Returns ({category: count, ..., "__other__": count}, (epsilon_spent, delta_spent)).
    """
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    # MO=L2Distance makes the histogram report L2 sensitivity, which then_gaussian needs.
    hist = space >> dp.t.then_count_by_categories(categories=categories, MO=dp.L2Distance[int])

    def make(scale):
        return dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(hist >> dp.m.then_gaussian(scale)), delta=delta)

    meas = dp.binary_search_chain(make, d_in=2, d_out=(epsilon, delta))   # replace-one = symmetric distance 2
    counts = meas(df[column].astype(str).tolist())   # last entry counts values not in categories
    return dict(zip(list(categories) + ["__other__"], counts)), meas.map(2)


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 40 + ["b"] * 35 + ["c"] * 20 + ["z"] * 5})
    _, (eps, _) = dp_histogram_gaussian(df, "group", ["a", "b", "c"], epsilon=1.0, delta=1e-6)
    return {"epsilon": float(eps)}
