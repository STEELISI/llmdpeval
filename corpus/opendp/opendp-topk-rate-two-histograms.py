import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


def dp_top_k_by_rate(df, column, candidates, k, epsilon, outcome_column, outcome_value):
    """Top-k public candidates by the RATE of rows with outcome_value, pure DP, replace-one
    neighbours (bounded DP, n public).
    candidates: PUBLIC list of every possible value, declared before looking at the data (codebook,
    schema), in a fixed order; never df[column].unique() (which values occur is itself private).

    Two OpenDP histograms (count_by_categories + Laplace) at epsilon/2 each: outcome rows per group
    and all rows per group. rate = noisy_num / noisy_den is post-processing.
    Returns ({"top": [...], "noisy_rates": {...}}, epsilon_spent).
    """
    space = (dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance())

    def make(scale):
        # extra null bin catches other values; a changed row moves between two bins (L1 2)
        return space >> dp.t.then_count_by_categories(categories=list(candidates)) >> dp.m.then_laplace(scale=scale)

    scale = dp.binary_search_param(make, d_in=2, d_out=epsilon / 2)   # replace-one = d_in 2; epsilon/2 each
    hit = (df[outcome_column] == outcome_value).to_numpy()
    groups = df[column].astype(str).to_numpy()
    m_num, m_den = make(scale), make(scale)
    noisy_num = np.array(m_num(groups[hit].tolist())[:-1], dtype=float)   # drop the null bin
    noisy_den = np.array(m_den(groups.tolist())[:-1], dtype=float)
    epsilon_spent = m_num.map(2) + m_den.map(2)       # each row is in both inputs: compose
    assert epsilon_spent <= epsilon + 1e-9
    # Post-processing: drop groups whose NOISY size is below 10 noise scales (fixed rule).
    rate = np.where(noisy_den >= 10 * scale, noisy_num / np.maximum(noisy_den, 1.0), -np.inf)
    order = [i for i in np.argsort(-rate) if np.isfinite(rate[i])][:k]   # groups below the cut-off are never picked
    return {"top": [candidates[i] for i in order],
            "noisy_rates": dict(zip(candidates, rate.tolist()))}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 300 + ["b"] * 300 + ["c"] * 300 + ["d"] * 3,
                       "flag": (["y"] * 240 + ["n"] * 60) + (["y"] * 90 + ["n"] * 210)
                               + (["y"] * 150 + ["n"] * 150) + ["y"] * 3})
    res, eps = dp_top_k_by_rate(df, "group", ["a", "b", "c", "d"], k=2, epsilon=1.0,
                                outcome_column="flag", outcome_value="y")
    assert len(res["top"]) == 2
    return {"epsilon": float(eps)}
