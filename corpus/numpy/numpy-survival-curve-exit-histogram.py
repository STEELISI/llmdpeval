import numpy as np


def dp_survival_curves(df, time_column, event_column, edges, epsilon, group_column=None, groups=None,
                       relation="replace_one", min_at_risk=1.0):
    """DP survival curves (Kaplan-Meier style, one point per PUBLIC time bin) per group of a PUBLIC group list.

    Each person EXITS once: in one (group, time bin, event-or-censored) cell. Release that one histogram,
    then rebuild at-risk counts, hazards and survival by cumulative sums (post-processing).
    edges: PUBLIC bin edges from 0 to the public maximum follow-up. Returns ({group: {...}}, epsilon_spent).
    """
    rng = np.random.default_rng()
    groups = ["all"] if group_column is None else list(groups)
    g = np.zeros(len(df), int) if group_column is None else df[group_column].map({k: i for i, k in enumerate(groups)}).to_numpy()
    t = df[time_column].to_numpy(dtype=float)
    ev = (df[event_column].to_numpy() == 1).astype(int)     # 1 = event; any other value counts as censored
    ev[t >= edges[-1]] = 0                                   # past the public end of follow-up: censored there
    B = len(edges) - 1
    b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, B - 1)
    keep = ~np.isnan(g.astype(float))                       # rows outside the public group list are dropped
    cell = (g[keep].astype(int) * B + b[keep]) * 2 + ev[keep]
    counts = np.bincount(cell, minlength=len(groups) * B * 2).astype(float)
    # One person sits in exactly one cell: L1 sensitivity 1 (add/remove), 2 (replace-one: the row moves).
    # Releasing at-risk counts directly would cost B times more: a person is at risk in every bin up to exit.
    l1 = 1.0 if relation == "add_remove" else 2.0
    scale = l1 / epsilon
    noisy = np.maximum(counts + rng.laplace(0.0, scale, size=counts.size), 0.0).reshape(len(groups), B, 2)
    epsilon_spent = l1 / scale
    assert epsilon_spent <= epsilon + 1e-9
    out = {}
    for i, name in enumerate(groups):                       # post-processing only
        events, censored = noisy[i, :, 1], noisy[i, :, 0]
        exits = events + censored
        at_risk = exits.sum() - np.concatenate([[0.0], np.cumsum(exits)[:-1]])   # start size is noisy too
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
    return {"epsilon": eps}
