import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


def dp_group_rates_parity(df, group_column, groups, outcome_column, outcome_value, group_a, group_b,
                          epsilon, min_size=1.0):
    """Per-group outcome rates and the statistical parity difference rate[a] - rate[b], pure DP, replace-one
    neighbours (bounded DP, n public); group sizes are private and come from a noisy histogram.

    groups: PUBLIC list of every group value (str), declared before looking at the data; never unique().
    min_size: PUBLIC floor for the noisy group size. Returns ({"rates", "noisy_sizes", "spd"}, epsilon_spent).
    """
    space = (dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance())

    def make(scale):
        # last bin = values outside the list; a changed row moves between two bins
        return space >> dp.t.then_count_by_categories(categories=list(groups)) >> dp.m.then_laplace(scale=scale)

    scale = dp.binary_search_param(make, d_in=2, d_out=epsilon / 2)   # replace-one = d_in 2; two histograms
    m_pos, m_all = make(scale), make(scale)
    g = df[group_column].astype(str)
    hit = (df[outcome_column] == outcome_value).to_numpy()
    noisy_pos = np.array(m_pos(g[hit].tolist())[:-1], dtype=float)    # drop the null bin
    noisy_all = np.array(m_all(g.tolist())[:-1], dtype=float)
    epsilon_spent = m_pos.map(2) + m_all.map(2)                        # sequential composition
    assert epsilon_spent <= epsilon + 1e-9
    # Post-processing: only released counts are used from here on.
    rates = dict(zip(groups, np.clip(noisy_pos / np.maximum(noisy_all, min_size), 0.0, 1.0).tolist()))
    return {"rates": rates, "noisy_sizes": dict(zip(groups, noisy_all.tolist())),
            "spd": rates[group_a] - rates[group_b]}, epsilon_spent


def _test():
    import pandas as pd
    GROUPS = ["g1", "g2", "g3"]   # public list, declared up front
    df = pd.DataFrame({"grp": ["g1"] * 200 + ["g2"] * 200, "pred": [1, 0] * 100 + [1, 0, 0, 0] * 50})
    res, eps = dp_group_rates_parity(df, "grp", GROUPS, "pred", 1, "g1", "g2", epsilon=1.0)
    assert -1.0 <= res["spd"] <= 1.0
    return {"epsilon": float(eps)}
