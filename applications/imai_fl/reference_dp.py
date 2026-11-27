"""Reference DP version of original.py, analyst side (Table 1 and Table 2 from the voter file).

Privacy unit: one voter (one row of voters.parquet). Replace-one (bounded) DP; only the total number of voters
(9,247,810) is public, every group size (race, precinct, district) is private. Total epsilon = 1, delta = 0,
basic sequential composition over the releases in BUDGET. Framework: OpenDP 0.16.0; every epsilon is read
from measurement.map(d_in) and asserted.

Public inputs (not private): p_* (Census surname list), r_prc_* / r_blk_* (2010 Census tables),
r_pid_* (wru's bundled table), the precinct and district ids, the 5 race labels. So each voter's predicted
probabilities are a per-record computation from public tables and that voter's own record; only the
aggregates below are released.

Releases
1. Table 1: for each of the 5 predictions, one histogram of voters by (self-reported race or "none") x
   (predicted class: 5 races, "no class" = tie, "unclassified" = no Census data): 6 x 7 counts.
   A voter is in one cell, so replace-one changes 2 cells: d_in = 2 (L1). Discrete Laplace.
2. Table 2: one vector of sums per (district, precinct) part (9,931 parts): voters and voters who voted by
   self-reported race (6 groups incl. "none"; one-hot), and for the name-only and the Bayesian prediction the
   sums of Pr(race) and of Pr(race) * voted (5 each). One voter adds at most 1 + 1 + 1.01 + 1.01 + 1 + 1 = 6.02
   to the L1 norm of the whole vector (the name-only row sum is at most 1.008 in the Census surname table;
   rows are clipped to L1 <= 1.01 to make that a guarantee), and the parts are disjoint, so replace-one
   changes the vector by at most d_in = 2 * 6.02 in L1 (parallel composition over parts, one release).
   Laplace on floats. Everything in Table 2 (observed turnout, Goodman, name-only, Bayesian; precinct and
   district rows) is post-processing of this vector: counts clipped at 0, turnout ratios clipped to [0, 1],
   "table2" is this uncorrected result (the grading bar). "table2_simex" adds SIMEX (see
   simex()), which removes part of the bias the noise adds to RMSEs and to Goodman's slopes: an expert ceiling.

Usage: python reference_dp.py <data_dir>
"""
import json
import sys

import numpy as np
import pandas as pd
import opendp.prelude as dp

import original as og

dp.enable_features("contrib")
EPS_TOTAL = 1.0
BUDGET = {f"table1_{k}": 0.02 for k in og.PREDICTIONS} | {"table2_parts": 0.9}
assert abs(sum(BUDGET.values()) - EPS_TOTAL) < 1e-12
NAME_L1 = 1.01                      # public bound on a voter's name-only probability row sum
T2_DIN = 2 * (1 + 1 + NAME_L1 + NAME_L1 + 1 + 1)
N_CLS = 7                           # 5 races, no class (tie), unclassified


def exact_stats(df):
    """Per-voter predictions (public tables + own record) -> exact Table 1 histograms and Table 2 part sums."""
    sr = df[og.SR].to_numpy(float)
    sr6 = np.hstack([sr, 1 - sr.sum(axis=1, keepdims=True)])          # 6th group: no self-reported race
    assert (sr6.min() >= 0) and (sr6.sum(axis=1) == 1).all()
    hists = {}
    preds = {k: og.predict(df, g, p) for k, (g, p) in og.PREDICTIONS.items()}
    for k, pr in preds.items():
        cls = og.classify(pr)
        col = np.where(np.isnan(cls[:, 0]), 6, np.where(np.nan_to_num(cls).sum(axis=1) == 0, 5,
                                                         np.nan_to_num(cls).argmax(axis=1)))
        hists[k] = np.bincount(sr6.argmax(axis=1) * N_CLS + col, minlength=6 * N_CLS)
    name = preds["name"]
    name = name * np.minimum(1, NAME_L1 / name.sum(axis=1, keepdims=True))  # no-op on this data
    bayes = np.nan_to_num(preds["name_precinct_party"])
    v = df["vote08"].to_numpy(float)[:, None]
    codes, parts = pd.MultiIndex.from_frame(df[["District", "Precinct"]]).factorize()
    X = np.hstack([sr6, sr6 * v, name, name * v, bayes, bayes * v])
    assert np.abs(X).sum(axis=1).max() <= T2_DIN / 2 + 1e-9
    sums = og.group_sum(codes, len(parts), X)
    return hists, sums, parts


def table1_from_hists(hists, n):
    """Table 1 error rates from the (noisy) confusion histograms; n is the public number of voters."""
    out = {}
    for k, h in hists.items():
        h = np.maximum(np.asarray(h, float), 0).reshape(6, N_CLS)
        pos = h[:5].sum(axis=1)
        diag = np.diag(h[:5, :5])
        classified = h[:, :6].sum()
        # error.R counts a voter with no self-reported race and no class (tie) as correct
        t = {"overall_error": 1 - (diag.sum() + h[5, 5]) / classified}
        for i, race in enumerate(og.RACE):
            t[f"{race}_fn"] = (pos[i] - diag[i] - h[i, 6]) / pos[i]
            t[f"{race}_fp"] = (h[:, i].sum() - diag[i]) / (n - pos[i])
        out[k] = {a: float(b) for a, b in t.items()}
    return out


def table2_from_parts(sums, parts):
    """Turnout.R / Goodman.R / Table2.R computed from the part sums only."""
    s = np.maximum(sums, 0)
    n6, v6 = s[:, 0:6], s[:, 6:12]
    est_sums = {"name": (s[:, 12:17], s[:, 17:22]), "bayes": (s[:, 22:27], s[:, 27:32])}
    d_ids = parts.get_level_values(0).to_numpy()
    p_codes, _ = pd.factorize(parts.get_level_values(1), sort=True)
    d_codes, _ = pd.factorize(d_ids, sort=True)
    n_p, n_d = p_codes.max() + 1, d_codes.max() + 1
    clip = lambda x: np.clip(x, 0, 1)
    n_prc, n_cd = og.group_sum(p_codes, n_p, n6[:, :5]), og.group_sum(d_codes, n_d, n6[:, :5])
    obs_prc = clip(og.ratio(og.group_sum(p_codes, n_p, v6[:, :5]), n_prc))
    obs_cd = clip(og.ratio(og.group_sum(d_codes, n_d, v6[:, :5]), n_cd))
    # Goodman: per district, OLS over parts of turnout on race share; precinct gets the n_k-weighted
    # average over its parts (the authors' de-duplication of split precincts).
    n_all = n6.sum(axis=1)
    vote = clip(og.ratio(v6.sum(axis=1), n_all))
    share = og.ratio(n6[:, :5], n_all[:, None])
    good_cd = np.full((n_d, 5), np.nan)
    for d in range(n_d):
        m = (d_codes == d) & (n_all > 0)
        for k in range(5):
            x, y = share[m, k], vote[m]
            if x.var() > 0:
                b = np.cov(x, y, bias=True)[0, 1] / x.var()
                good_cd[d, k] = y.mean() + b * (1 - x.mean())
    est_part = good_cd[d_codes]
    good_prc = og.ratio(og.group_sum(p_codes, n_p, n6[:, :5] * est_part), n_prc)
    single = np.bincount(p_codes, minlength=n_p) == 1
    idx = np.flatnonzero(single[p_codes])
    good_prc[p_codes[idx]] = est_part[idx]
    est = {"goodman": (good_prc, good_cd)}
    for k, (t, sv) in est_sums.items():
        est[k] = (clip(og.ratio(og.group_sum(p_codes, n_p, sv), og.group_sum(p_codes, n_p, t))),
                  clip(og.ratio(og.group_sum(d_codes, n_d, sv), og.group_sum(d_codes, n_d, t))))
    rows = {f"{lvl}_{r}": {} for lvl in ["precinct", "district"] for r in og.RACE}
    for k, (e_p, e_d) in est.items():
        b_p, r_p = og.bias_rmse_weighted(e_p, obs_prc, n_prc)
        b_d, r_d = og.bias_rmse_mean(e_d, obs_cd)
        for i, r in enumerate(og.RACE):
            rows[f"precinct_{r}"].update({f"{k}_bias": float(b_p[i]), f"{k}_rmse": float(r_p[i])})
            rows[f"district_{r}"].update({f"{k}_bias": float(b_d[i]), f"{k}_rmse": float(r_d[i])})
    return rows


def simex(noisy_sums, parts, scale, lams=(1.5, 2.0, 2.5, 3.0), reps=10):
    """SIMEX (Cook & Stefanski, JASA 1994) on the released sums: add more noise so the noise variance is
    lam times the release's, recompute Table 2, fit a quadratic in lam per cell and read it at lam = 0.
    Uses only the DP output and the public noise scale, so it is post-processing (no privacy cost).
    It removes most of the RMSE inflation that noise in the observed precinct turnout causes, and the
    attenuation of Goodman's slopes from noisy race shares."""
    rng = np.random.default_rng()
    flat = lambda t: np.array([[t[r][c] for c in sorted(t[r])] for r in sorted(t)])
    base = table2_from_parts(noisy_sums, parts)
    xs, ys = [1.0], [flat(base)]
    for lam in lams:
        sims = [flat(table2_from_parts(noisy_sums + rng.laplace(0, scale * np.sqrt(lam - 1), noisy_sums.shape),
                                       parts)) for _ in range(reps)]
        xs.append(lam)
        ys.append(np.nanmean(sims, axis=0))
    Y = np.stack(ys)                                            # lam x rows x cols
    coef = np.polyfit(xs, Y.reshape(len(xs), -1), 2)            # quadratic in lam, per cell
    at0 = coef[2].reshape(Y.shape[1:])
    rows, cols = sorted(base), sorted(base[sorted(base)[0]])
    out = {r: {c: float(at0[i, j]) for j, c in enumerate(cols)} for i, r in enumerate(rows)}
    for r in out:                                               # an RMSE cannot be negative
        for c in out[r]:
            if c.endswith("rmse"):
                out[r][c] = max(out[r][c], 0.0)
    return out


def release(exact, n):
    """Noise the exact statistics with OpenDP and post-process them into Table 1 and Table 2; returns the tables and
    epsilon spent.
    """
    hists, sums, parts = exact
    spent, noisy_h = {}, {}
    int_space = dp.vector_domain(dp.atom_domain(T=int)), dp.l1_distance(T=int)
    for k, h in hists.items():
        m = dp.binary_search_chain(lambda s: dp.m.make_laplace(*int_space, scale=s), d_in=2,
                                   d_out=BUDGET[f"table1_{k}"])
        spent[f"table1_{k}"] = m.map(2)
        noisy_h[k] = m(h.astype(int).tolist())
    flt_space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.l1_distance(T=float)
    scale = dp.binary_search_param(lambda s: dp.m.make_laplace(*flt_space, scale=s), d_in=T2_DIN,
                                   d_out=BUDGET["table2_parts"])
    m = dp.m.make_laplace(*flt_space, scale=scale)
    spent["table2_parts"] = m.map(T2_DIN)
    noisy_sums = np.array(m(sums.ravel().tolist())).reshape(sums.shape)
    for k, e in spent.items():
        assert e <= BUDGET[k] + 1e-9, (k, e)
    assert sum(spent.values()) <= EPS_TOTAL + 1e-9, spent
    return {"table1": table1_from_hists(noisy_h, n), "table2": table2_from_parts(noisy_sums, parts),
            "table2_simex": simex(noisy_sums, parts, scale),  # expert ceiling only, not the grading bar
            "epsilon_spent": {k: float(v) for k, v in spent.items()}, "epsilon_total": float(sum(spent.values()))}


if __name__ == "__main__":
    df = og.load(sys.argv[1])
    ex = exact_stats(df)
    true = {"table1": table1_from_hists(ex[0], len(df)), "table2": table2_from_parts(ex[1], ex[2])}
    out = release(ex, len(df))
    print("Table 1, true vs DP:")
    print(pd.concat({"true": pd.DataFrame(true["table1"]), "dp": pd.DataFrame(out["table1"])}, axis=1).round(3).to_string())
    print("\nTable 2, true vs DP:")
    print(pd.concat({"true": pd.DataFrame(true["table2"]).T, "dp": pd.DataFrame(out["table2"]).T}, axis=1).round(3).to_string())
    print(f"\nepsilon spent {out['epsilon_spent']}  total {out['epsilon_total']:.4f} (budget {EPS_TOTAL}, delta 0)")
    print(json.dumps({"true": true, "dp": out}))
