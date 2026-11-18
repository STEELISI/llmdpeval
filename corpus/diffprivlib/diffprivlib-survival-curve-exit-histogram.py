import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import histogram


def dp_survival_curves(df, time_column, event_column, edges, epsilon, group_column=None, groups=None,
                       min_at_risk=1.0):
    """DP survival curves (Kaplan-Meier style, one point per PUBLIC time bin) per group of a PUBLIC group list.
    Replace-one neighbours (bounded DP, n public).

    Each person EXITS once: in one (group, time bin, event-or-censored) cell. One diffprivlib histogram over
    the cell codes, then at-risk counts, hazards and survival by cumulative sums (post-processing).
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
    n_cells = len(groups) * B * 2
    acc = BudgetAccountant()
    # diffprivlib calibrates for add/remove (one bin moves by 1). Under replace-one the row moves between two
    # cells, so pass epsilon/2; the guarantee is 2x the accountant total. Never release at-risk counts directly:
    # a person is at risk in every bin up to exit (sensitivity B).
    counts, _ = histogram(cell, epsilon=epsilon / 2, bins=n_cells, range=(-0.5, n_cells - 0.5), accountant=acc)
    epsilon_spent = 2 * acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    noisy = counts.astype(float).reshape(len(groups), B, 2)
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
