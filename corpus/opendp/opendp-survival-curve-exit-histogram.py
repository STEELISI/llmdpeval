import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


def dp_survival_curves(df, time_column, event_column, edges, epsilon, group_column=None, groups=None,
                       min_at_risk=1.0):
    """DP survival curves (Kaplan-Meier style, one point per PUBLIC time bin) per group of a PUBLIC group list.
    Replace-one neighbours (bounded DP, n public): evaluated at d_in = 2.

    Each person EXITS once: in one (group, time bin, event-or-censored) cell. One OpenDP histogram over the cell
    codes, then at-risk counts, hazards and survival by cumulative sums (post-processing).
    edges: PUBLIC bin edges from 0 to the public maximum follow-up. Returns ({group: {...}}, epsilon_spent).
    """
    groups = ["all"] if group_column is None else list(groups)
    g = np.zeros(len(df)) if group_column is None else df[group_column].map({k: i for i, k in enumerate(groups)}).to_numpy(dtype=float)
    t = df[time_column].to_numpy(dtype=float)
    ev = df[event_column].to_numpy(dtype=int).copy()
    ev[t >= edges[-1]] = 0                                   # past the public end of follow-up: censored there
    B = len(edges) - 1
    b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, B - 1)
    keep = ~np.isnan(g)                                      # rows outside the public group list are dropped
    cell = (g[keep].astype(int) * B + b[keep]) * 2 + ev[keep]
    cats = list(range(len(groups) * B * 2))
    space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
    # One cell per person; the count vector moves by 2 in L1 when a row is replaced. Never release at-risk
    # counts directly: a person is at risk in every bin up to exit (sensitivity B).
    make = lambda s: space >> dp.t.then_count_by_categories(categories=cats, null_category=False) >> dp.m.then_laplace(s)
    meas = dp.binary_search_chain(make, d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)
    assert epsilon_spent <= epsilon + 1e-9
    noisy = np.maximum(np.array(meas(cell.tolist()), dtype=float), 0.0).reshape(len(groups), B, 2)
    out = {}
    for i, name in enumerate(groups):                       # post-processing only
        events, exits = noisy[i, :, 1], noisy[i].sum(axis=1)
        at_risk = exits.sum() - np.concatenate([[0.0], np.cumsum(exits)[:-1]])
        hazard = np.clip(events / np.maximum(at_risk, min_at_risk), 0.0, 1.0)
        out[name] = {"at_risk": at_risk, "events": events, "hazard": hazard, "survival": np.cumprod(1 - hazard)}
    return out, epsilon_spent


def _test():
    import pandas as pd
    rng = np.random.default_rng()
    n = 2000
    df = pd.DataFrame({"time": rng.exponential(5.0, n), "event": rng.integers(0, 2, n),
                       "grp": rng.choice(["a", "b"], n)})
    res, eps = dp_survival_curves(df, "time", "event", [float(e) for e in range(0, 11)], 1.0, "grp", ["a", "b", "c"])
    assert all(np.all(np.diff(v["survival"]) <= 1e-12) for v in res.values())
    return {"epsilon": float(eps)}
