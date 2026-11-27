"""Predicting individual race from voter records, and estimating turnout by race (Florida, 2008).

Python port of Imai & Khanna (Political Analysis 24(2) 2016): the race predictions of Predict.R
(wru 0.0-1 race.pred), the classification error rates of Table 1, and the turnout-by-race estimates of
Table 2 (Goodman's regression, name-only prediction, Bayesian prediction with name + precinct + party;
bias and RMSE against self-reported race at precinct and district level). King's EI is not ported.

Each voter's race probabilities come from Bayes' rule:
  Pr(race | surname, geo, party) is proportional to Pr(race | surname) * Pr(geo | race) * Pr(party | race).
p_* = Pr(race | surname) (Census surname list), r_prc_* / r_blk_* = Pr(precinct or block | race)
(2010 Census counts), r_pid_* = Pr(party | race) (wru's bundled table).

Usage: python original.py <data_dir>   (data_dir holds voters.parquet)
"""
import json
import sys

import numpy as np
import pandas as pd

ETH = ["whi", "bla", "his", "asi", "oth"]
SR = [f"SR.{e.upper()}" for e in ETH]
RACE = ["white", "black", "latino", "asian", "other"]
# Table 1 columns: (geography, use party)
PREDICTIONS = {"name": (None, False), "name_precinct": ("prc", False), "name_block": ("blk", False),
               "name_precinct_party": ("prc", True), "name_block_party": ("blk", True)}


def cols(df, prefix):
    """The five race columns with the given prefix, as a float array."""
    return df[[f"{prefix}_{e}" for e in ETH]].to_numpy(float)


def predict(df, geo=None, party=False, r_geo=None):
    """wru 0.0-1 race.pred with census = "": surname only, or p * r (* r_pid) normalised per voter.
    r_geo overrides the Pr(geo | race) columns (used for the Census-side DP experiment)."""
    P = cols(df, "p")
    if geo is None:
        return P  # race.pred(surname.only = TRUE) returns p_* unnormalised
    U = P * (cols(df, f"r_{geo}") if r_geo is None else r_geo)
    if party:
        U = U * cols(df, "r_pid")
    with np.errstate(invalid="ignore", divide="ignore"):
        return U / np.nansum(U, axis=1, keepdims=True)  # 0/0 -> NaN, as in R


def classify(pred):
    """classify.R: 1 for the race with the strictly highest probability, ties -> all 0, NaN rows stay NaN."""
    nan_row = np.isnan(pred).all(axis=1)
    assert not (np.isnan(pred).any(axis=1) & ~nan_row).any(), "partly missing predictions not handled"
    top = pred == np.nanmax(np.where(nan_row[:, None], 0, pred), axis=1, keepdims=True)
    cls = (top & (top.sum(axis=1, keepdims=True) == 1)).astype(float)
    cls[nan_row] = np.nan
    return cls


def table1_column(cls, sr):
    """error.R: overall error over classified voters; per race false negative / false positive rates.
    As in error.R, the denominators (all positives / all negatives) include unclassified voters."""
    ok = ~np.isnan(cls[:, 0])
    out = {"overall_error": float((cls[ok] != sr[ok]).any(axis=1).mean())}
    for k, race in enumerate(RACE):
        pos = sr[:, k] == 1
        out[f"{race}_fn"] = float((pos & ok & (cls[:, k] == 0)).sum() / pos.sum())
        out[f"{race}_fp"] = float((~pos & ok & (cls[:, k] == 1)).sum() / (~pos).sum())
    return out


def group_sum(codes, n_groups, values):
    """Column-wise sums of values (rows x 5) per group code, NaN treated as 0 (R's na.rm = TRUE)."""
    v = np.nan_to_num(values)
    return np.stack([np.bincount(codes, v[:, k], n_groups) for k in range(v.shape[1])], axis=1)


def ratio(num, den):
    """Elementwise num / den, with 0/0 -> NaN and x/0 -> inf without warnings."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def goodman(df, prc_codes, n_prc):
    """Goodman.R: per district, regress precinct turnout on the precinct's share of each race
    (one simple OLS per race); estimate = intercept + slope. Precinct estimate = its district's estimate;
    a precinct split over districts gets the average weighted by its voters of that race in each part."""
    sr = df[SR].to_numpy(float)
    part = pd.DataFrame({"d": df["District"].to_numpy(), "p": prc_codes, "v": df["vote08"].to_numpy(float)})
    part[ETH] = sr
    g = part.groupby(["d", "p"], sort=False)
    parts = g[["v"] + ETH].sum()
    n = g.size().to_numpy(float)
    vote = parts["v"].to_numpy() / n
    share = parts[ETH].to_numpy() / n[:, None]
    d_of = parts.index.get_level_values("d").to_numpy()
    p_of = parts.index.get_level_values("p").to_numpy()
    districts = np.sort(np.unique(d_of))
    est_cd = np.full((len(districts), 5), np.nan)
    for i, d in enumerate(districts):
        m = d_of == d
        for k in range(5):
            x, y = share[m, k], vote[m]
            if x.var() > 0:
                slope = np.cov(x, y, bias=True)[0, 1] / x.var()
                est_cd[i, k] = y.mean() - slope * x.mean() + slope
    est_part = est_cd[np.searchsorted(districts, d_of)]
    w = parts[ETH].to_numpy()
    est_prc = ratio(group_sum(p_of, n_prc, w * est_part), group_sum(p_of, n_prc, w))
    single = np.bincount(p_of, minlength=n_prc) == 1  # one district: keep its estimate even if 0 voters
    idx = np.flatnonzero(single[p_of])
    est_prc[p_of[idx]] = est_part[idx]
    return est_cd, est_prc


def bias_rmse_weighted(est, obs, w):
    """Table2.R precinct rows: weights = voters of that race; NaN terms dropped from the numerator only."""
    e = est - obs
    return (np.nansum(w * e, axis=0) / w.sum(axis=0),
            np.sqrt(np.nansum(w * e ** 2, axis=0) / w.sum(axis=0)))


def bias_rmse_mean(est, obs):
    """Per-race mean error and root mean squared error of est against obs, ignoring NaN."""
    e = est - obs
    return np.nanmean(e, axis=0), np.sqrt(np.nanmean(e ** 2, axis=0))


def table2(df, preds):
    """Turnout.R + Goodman.R + Table2.R. preds = {"name": ..., "bayes": ...} (voters x 5 probabilities)."""
    prc_codes, prc_ids = pd.factorize(df["Precinct"], sort=True)
    cd_codes, cd_ids = pd.factorize(df["District"], sort=True)
    n_prc, n_cd = len(prc_ids), len(cd_ids)
    sr = df[SR].to_numpy(float)
    v = df["vote08"].to_numpy(float)[:, None]
    nsr_prc, nsr_cd = group_sum(prc_codes, n_prc, sr), group_sum(cd_codes, n_cd, sr)
    obs_prc = ratio(group_sum(prc_codes, n_prc, sr * v), nsr_prc)
    obs_cd = ratio(group_sum(cd_codes, n_cd, sr * v), nsr_cd)
    good_cd, good_prc = goodman(df, prc_codes, n_prc)
    est = {"goodman": (good_prc, good_cd)}
    for name, pr in preds.items():
        est[name] = (ratio(group_sum(prc_codes, n_prc, pr * v), group_sum(prc_codes, n_prc, pr)),
                     ratio(group_sum(cd_codes, n_cd, pr * v), group_sum(cd_codes, n_cd, pr)))
    rows = {f"{level}_{race}": {} for level in ["precinct", "district"] for race in RACE}
    for name, (e_prc, e_cd) in est.items():
        b_p, r_p = bias_rmse_weighted(e_prc, obs_prc, nsr_prc)
        b_d, r_d = bias_rmse_mean(e_cd, obs_cd)
        for k, race in enumerate(RACE):
            rows[f"precinct_{race}"].update({f"{name}_bias": float(b_p[k]), f"{name}_rmse": float(r_p[k])})
            rows[f"district_{race}"].update({f"{name}_bias": float(b_d[k]), f"{name}_rmse": float(r_d[k])})
    return rows


def load(data_dir):
    """Read voters.parquet from data_dir."""
    return pd.read_parquet(f"{data_dir}/voters.parquet")


def main(data_dir):
    df = load(data_dir)
    sr = df[SR].to_numpy(float)
    print(f"voters {len(df)}, precincts {df['Precinct'].nunique()}, districts {df['District'].nunique()}")
    print("self-reported race shares:", dict(zip(RACE, sr.mean(axis=0).round(3))))
    preds = {name: predict(df, geo, party) for name, (geo, party) in PREDICTIONS.items()}
    t1 = {name: table1_column(classify(p), sr) for name, p in preds.items()}
    print("\nTable 1: classification error rates")
    print(pd.DataFrame(t1).round(3).to_string())
    t2 = table2(df, {"name": preds["name"], "bayes": preds["name_precinct_party"]})
    print("\nTable 2: turnout by race, bias and RMSE (goodman / name-only / bayes = name + precinct + party)")
    print(pd.DataFrame(t2).T.round(3).to_string())
    print(json.dumps({"table1": t1, "table2": t2}))


if __name__ == "__main__":
    main(sys.argv[1])
