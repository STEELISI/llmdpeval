"""Labor unions and racial resentment among white Americans (CCES 2010 and 2012).

Regress the racial resentment index on current and past union membership, for white
respondents, in three models: bivariate; with demographic controls and state fixed
effects; and with year fixed effects added. Also report the mean racial resentment of
union members and non-union members. Rows with a missing value in a model's variables
are dropped from that model.

Usage: python original.py <data_dir>   (data_dir holds cces_white.csv)
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

Y = "racial_resentment"
MODELS = {
    1: ["union_member", "past_union_member"],
    2: ["union_member", "past_union_member", "female", "famincome", "age", "educ", "state"],
    3: ["union_member", "past_union_member", "female", "famincome", "age", "educ", "year", "state"],
}
RAW = {"female": "gender", "state": "state_pre"}  # derived regressor -> raw column


def load(data_dir):
    """Read cces_white.csv from data_dir."""
    return pd.read_csv(f"{data_dir}/cces_white.csv")


def design(df, terms):
    """Complete cases for the model, then X with an intercept. Gender, year and state enter as
    dummies with the lowest code as the reference level."""
    raw = [Y] + [RAW.get(t, t) for t in terms]
    d = df.dropna(subset=raw)
    cols = {"constant": np.ones(len(d))}
    for t in terms:
        if t == "female":
            cols["female"] = (d["gender"] == 2).astype(float).to_numpy()
        elif t in ("year", "state"):
            raw_col = d[RAW.get(t, t)]
            for level in sorted(raw_col.unique())[1:]:
                cols[f"{t}_{level}"] = (raw_col == level).astype(float).to_numpy()
        else:
            cols[t] = d[t].astype(float).to_numpy()
    return pd.DataFrame(cols, index=d.index), d[Y].to_numpy(float)


def ols(X, y):
    """Least squares with classical standard errors and a two-sided t test."""
    Xv = X.to_numpy()
    n, k = Xv.shape
    beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
    resid = y - Xv @ beta
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(sigma2 * np.diag(np.linalg.inv(Xv.T @ Xv)))
    p = 2 * stats.t.sf(np.abs(beta / se), n - k)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return pd.DataFrame({"coef": beta, "se": se, "p": p}, index=X.columns), n, r2


def group_means(df):
    """Mean racial resentment of union members vs non-members (rows of model 1)."""
    d = df.dropna(subset=[Y, "union_member", "past_union_member"])
    g = d.groupby("union_member")[Y]
    m, v, n = g.mean(), g.var(), g.size()
    diff = m[1.0] - m[0.0]
    se = np.sqrt(v[1.0] / n[1.0] + v[0.0] / n[0.0])
    return {"mean_union": m[1.0], "mean_nonunion": m[0.0], "diff": diff, "se": se,
            "p": 2 * stats.norm.sf(abs(diff / se))}


def main(data_dir):
    df = load(data_dir)
    out = {}
    for k, terms in MODELS.items():
        res, n, r2 = ols(*design(df, terms))
        shown = [t for t in res.index if not t.startswith(("state_", "year_"))]
        print(f"Model {k}: N = {n}, R2 = {r2:.3f}")
        print(res.loc[shown].round(4).to_string())
        out[f"model{k}"] = {"N": n, "r2": r2, **{t: res.loc[t, ["coef", "se", "p"]].tolist() for t in shown}}
    gm = group_means(df)
    print(f"mean racial resentment: union {gm['mean_union']:.4f}, non-union {gm['mean_nonunion']:.4f}, "
          f"difference {gm['diff']:.4f} (SE {gm['se']:.4f}, p = {gm['p']:.2g})")
    out["group_means"] = gm
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
