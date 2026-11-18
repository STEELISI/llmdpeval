import opendp.prelude as dp

dp.enable_features("contrib")


def dp_select_candidate_above_threshold(df, column, bounds, threshold, stop_probability, epsilon):
    """Private selection (Liu and Talwar 2019, Alg. 1): rerun a scored DP measurement until its noisy
    score passes a public threshold, and release that (score, candidate) pair. The wrapper costs
    twice the inner measurement's epsilon. Replace-one (n public): sized domain, d_in = 2
    (add/remove-one, n private: unsized, d_in = 1).

    Here the score is a DP count and the candidate a DP sum; any measurement returning a
    (score, candidate) pair works. Returns ((score, candidate) or None, epsilon_spent).
    """
    values = df[column].astype(float).tolist()
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()
    inner_eps = epsilon / 4          # score + candidate = epsilon/2; selection doubles it
    m_score = dp.binary_search_chain(lambda s: space >> dp.t.then_count() >> dp.m.then_laplace(s),
                                     d_in=2, d_out=inner_eps)
    m_cand = dp.binary_search_chain(lambda s: space >> dp.t.then_clamp(bounds) >> dp.t.then_sum()
                                    >> dp.m.then_laplace(s), d_in=2, d_out=inner_eps)
    m_scored = dp.c.make_composition([m_score, m_cand])     # releases [score, candidate]

    # stop_probability > 0 guarantees termination when the threshold is never reached.
    m_select = dp.c.make_select_private_candidate(m_scored, stop_probability=stop_probability,
                                                  threshold=threshold)
    released = m_select(values)   # None if it stopped early
    return released, m_select.map(2)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, epsilon_spent = dp_select_candidate_above_threshold(df, "value", (-5.0, 5.0), threshold=0.0,
                                                           stop_probability=0.05, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
