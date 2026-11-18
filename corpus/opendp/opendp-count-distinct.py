import opendp.prelude as dp

dp.enable_features("contrib")


def dp_count_distinct(df, column, epsilon):
    """Number of distinct values in a column.

    Replace-one (n public): changing one row moves the distinct count by at most 1 (a value may
    vanish and another appear; the net change is -1, 0 or +1). OpenDP's d_in=1 map of then_count_distinct is exactly 1,
    the replace-one sensitivity; its d_in=2 map charges 2 (conservative). So d_in=1 is calibrated
    for replace-one here, and also for add/remove-one. Histograms and group-bys need d_in=2.

    Returns (private_distinct_count, epsilon_spent).
    """
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    d_in = 1                                            # see docstring: = replace-one sensitivity 1
    meas = dp.binary_search_chain(
        lambda s: space >> dp.t.then_count_distinct() >> dp.m.then_laplace(s), d_in=d_in, d_out=epsilon)
    dp_value = meas(df[column].astype(str).tolist())    # integer output, discrete Laplace noise
    return dp_value, meas.map(d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"key": [f"k{i % 37}" for i in range(300)]})
    _, epsilon_spent = dp_count_distinct(df, "key", epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
