import opendp.prelude as dp

dp.enable_features("contrib")


def dp_quantiles_via_histogram(df, column, edges, alphas, epsilon):
    """Many DP quantiles from one noisy histogram over public bin edges.
    Replace-one neighbours (bounded DP, n public): symmetric distance d_in = 2.

    Only the Laplace step spends budget; the quantiles are post-processing.
    Returns (list_of_quantiles, epsilon_spent).
    """
    bins = [str(i) for i in range(len(edges) - 1)]
    space = dp.vector_domain(dp.atom_domain(T=float, nan=True)), dp.symmetric_distance()
    hist = (space
            >> dp.t.then_drop_null()
            >> dp.t.then_find_bin(edges=edges[1:-1])   # interior cut points -> bin index
            >> dp.t.then_index(bins, "0")
            >> dp.t.then_count_by_categories(categories=bins, null_category=False))

    def make(scale):
        return (hist >> dp.m.then_laplace(scale)
                >> dp.t.make_cast_default(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance(), TOA=float)
                >> dp.t.make_quantiles_from_counts(edges, alphas=alphas))

    # A replaced row leaves one bin and enters another: the count vector moves by 2 in L1.
    meas = dp.binary_search_chain(make, d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)
    assert epsilon_spent <= epsilon + 1e-9
    return meas(df[column].astype(float).tolist()), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 50) for i in range(500)]})
    edges = [float(e) for e in range(0, 55, 5)]
    qs, epsilon_spent = dp_quantiles_via_histogram(df, "value", edges, [0.25, 0.5, 0.75], 1.0)
    assert len(qs) == 3
    return {"epsilon": float(epsilon_spent)}
