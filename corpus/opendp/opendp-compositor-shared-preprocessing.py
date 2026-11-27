import opendp.prelude as dp

dp.enable_features("contrib")


def dp_shared_preprocessing_queries(df, column, bounds, impute_value, epsilon):
    """Chain one preprocessing pipeline (parse text -> impute -> clamp) in front of an adaptive
    compositor, so every query reuses it. Replace-one on the raw rows (n public): sized domain,
    raw d_in = 2. Alternative, add/remove-one (n private): unsized domain, raw d_in = 1.

    The column holds text; bounds and impute_value are public. Returns ((count, sum), epsilon_spent).
    """
    values = df[column].astype(str).tolist()
    raw_space = dp.vector_domain(dp.atom_domain(T=str), size=len(values)), dp.symmetric_distance()
    prep = (raw_space
            >> dp.t.then_cast_default(float)        # unparseable text becomes 0.0 (no exceptions)
            >> dp.t.then_impute_constant(impute_value)
            >> dp.t.then_clamp(bounds))
    # The compositor's d_in is the distance AFTER preprocessing: ask the transformation.
    d_mid = prep.map(2)                       # raw replace-one distance 2 -> 2 after preprocessing
    meas_comp = prep >> dp.c.then_adaptive_composition(
        output_measure=dp.max_divergence(), d_in=d_mid, d_mids=[epsilon / 2, epsilon / 2])
    qbl = meas_comp(values)

    # Queries start from the compositor's input space (the clamped float vectors).
    space = prep.output_space
    m_count = dp.binary_search_chain(lambda s: space >> dp.t.then_count() >> dp.m.then_laplace(s),
                                     d_in=d_mid, d_out=epsilon / 2)
    m_sum = dp.binary_search_chain(lambda s: space >> dp.t.then_sum() >> dp.m.then_laplace(s),
                                   d_in=d_mid, d_out=epsilon / 2)
    result = (qbl(m_count), qbl(m_sum))
    return result, meas_comp.map(2)


def _test():
    import pandas as pd
    df = pd.DataFrame({"raw": [str(i % 11 - 5) for i in range(99)] + ["n/a"]})
    _, epsilon_spent = dp_shared_preprocessing_queries(df, "raw", (-5.0, 5.0), 0.0, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
