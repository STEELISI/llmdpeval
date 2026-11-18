"""Reference DP version of original.py: Table 1 models 1-3 and the union / non-union means,
each with a noise-aware standard error and test.

Privacy unit: one white respondent (one row), replace-one (bounded) DP. Public: the model sizes
printed in Table 1 (71,431 rows for model 1, 61,463 for models 2 and 3) and the codebook domains
below. Group sizes (union members, states, years) are private. Total epsilon = 1, delta = 0,
basic sequential composition over two releases (see BUDGET):
  1. model 1 rows: histogram over union status (current / past / never) x the 9 resentment values,
     OpenDP count-by-categories + discrete Laplace. Model 1 and the group means are post-processing.
  2. model 2/3 rows: noisy sufficient statistics (column sums, cross-products and squares of the
     regressors scaled to [0, 1] by public bounds, and of y), OpenDP vector Laplace. Model 3 uses
     them all; model 2 drops the year entries (post-processing).
Inference: SE^2 = sampling variance (plug-in, from the noisy statistics) + DP-noise variance
(Monte Carlo: re-noise the released statistics with the known scale and refit). p-values use a
normal reference. Sources and limits in spec.md.

Framework: OpenDP 0.16.0. Usage: python reference_dp.py <data_dir>
"""
import json
import sys

import numpy as np
import opendp.prelude as dp
from scipy import stats

import original

dp.enable_features("contrib")
EPS_TOTAL = 1.0
BUDGET = {"histogram_model1": 0.2, "moments_model23": 0.8}
assert abs(sum(BUDGET.values()) - EPS_TOTAL) < 1e-12
N1, N23 = 71431, 61463  # printed in Table 1 (public)
D_IN = 2  # replace-one on a sized dataset = symmetric distance 2 in OpenDP
MC_DRAWS = 500  # Monte Carlo draws for the DP-noise variance

# Public domains (spec.md gives the sources; values outside are clamped).
LEVELS = np.arange(9) / 8  # racial_resentment: mean of two 5-point items on [0, 1]
STATUS = ["current", "past", "never"]  # from union_member / past_union_member
BOUNDS = {"famincome": (1, 18), "age": (18, 100), "educ": (1, 6)}  # age bound: spec.md rule 2
STATES = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56]  # FIPS
YEARS = [2010, 2012]
TERMS = ["constant", "union_member", "past_union_member", "female", "famincome", "age", "educ"]

# Model 2/3 regressors after scaling, all in [0, 1]; y is the last column.
COLS = (["union_member", "past_union_member", "female", "famincome", "age", "educ", "year_2012"]
        + [f"state_{s}" for s in STATES[1:]] + ["y"])
BINARY = {c for c in COLS if c not in ("famincome", "age", "educ", "y")}
# Released entries: every column sum, every cross-product of two different columns, and the squares
# of the non-binary columns. Left out because they are 0 or a duplicate for every possible row:
# squares of binary columns (= the sum), two different states, current x past union.
PAIRS = [(i, j) for i in range(len(COLS)) for j in range(i, len(COLS))
         if not (i == j and COLS[i] in BINARY)
         and not (i != j and COLS[i].startswith("state_") and COLS[j].startswith("state_"))
         and (COLS[i], COLS[j]) != ("union_member", "past_union_member")]
# Per-row L1 norm of [z, z_i z_j over PAIRS]: at most 8 non-zero z entries (one union status, female,
# 3 scaled numerics, year, one state, y), so 8 sums + 28 cross-products + 4 squares = 40.
ROW_L1 = 40
D_MOMENTS = 2.0 * ROW_L1  # replace-one: remove one row and add another


def p_value(est, se):
    """Two-sided p-value of est / se against a standard normal."""
    return float(2 * stats.norm.sf(abs(est / se)))


# ---------- release 1: model 1 and group means from a histogram ----------
def model1_rows(df):
    """Model 1 complete cases as 'status|level' labels (union status x resentment level 0..8)."""
    d = df.dropna(subset=[original.Y, "union_member", "past_union_member"])
    assert len(d) == N1, len(d)
    status = np.where(d["union_member"] == 1, "current", np.where(d["past_union_member"] == 1, "past", "never"))
    level = np.rint(np.clip(d[original.Y].to_numpy(), 0, 1) * 8).astype(int)
    return [f"{s}|{l}" for s, l in zip(status, level)]


def search(make, d_in, eps):
    """Smallest noise scale whose OpenDP privacy map gives <= eps; returns (measurement, scale)."""
    scale = dp.binary_search_param(make, d_in=d_in, d_out=eps)
    return make(scale), scale


def histogram_meas(eps):
    """OpenDP sized count-by-categories + discrete Laplace over the 27 model-1 cells, calibrated to eps; returns
    (measurement, scale, categories).
    """
    space = dp.vector_domain(dp.atom_domain(T=str), size=N1), dp.symmetric_distance()
    cats = [f"{s}|{l}" for s in STATUS for l in range(9)]
    m, scale = search(lambda s: space >> dp.t.then_count_by_categories(categories=cats, null_category=False)
                      >> dp.m.then_laplace(s), D_IN, eps)
    return m, scale, cats


def fit_histogram(cells):
    """cells: (..., 3, 9) counts. Returns model 1 (saturated in the 3 groups) and the union / non-union
    means, each with its plug-in sampling variance. Works on a batch of noisy histograms."""
    c = np.maximum(cells, 0)
    n_g = np.maximum(c.sum(-1), 1)
    s_g, q_g = c @ LEVELS, c @ LEVELS**2
    mean_g = s_g / n_g
    n_never = np.maximum(N1 - n_g[..., 0] - n_g[..., 1], 1)  # never = public n minus the two noisy groups
    ssr = (q_g - s_g**2 / n_g).sum(-1)
    sigma2 = np.maximum(ssr / (N1 - 3), 1e-12)
    sst = q_g.sum(-1) - s_g.sum(-1)**2 / N1
    mean_nonunion = (s_g[..., 1] + s_g[..., 2]) / (N1 - n_g[..., 0])
    var_nonunion = (q_g[..., 1] + q_g[..., 2]) / (N1 - n_g[..., 0]) - mean_nonunion**2
    return {
        "union_member": (mean_g[..., 0] - mean_g[..., 2], sigma2 * (1 / n_g[..., 0] + 1 / n_never)),
        "past_union_member": (mean_g[..., 1] - mean_g[..., 2], sigma2 * (1 / n_g[..., 1] + 1 / n_never)),
        "constant": (mean_g[..., 2], sigma2 / n_never),
        "r2": (1 - ssr / sst, None),
        "mean_union": (mean_g[..., 0], None), "mean_nonunion": (mean_nonunion, None),
        "diff": (mean_g[..., 0] - mean_nonunion,
                 (q_g[..., 0] / n_g[..., 0] - mean_g[..., 0]**2) / n_g[..., 0] + var_nonunion / (N1 - n_g[..., 0])),
    }


def discrete_laplace(scale, size, rng):
    """Discrete Laplace draws (difference of two geometrics). Used only to re-noise released counts for the Monte Carlo
    SE, never for the release itself.
    """
    p = 1 - np.exp(-1 / scale)
    return rng.geometric(p, size) - rng.geometric(p, size)


def histogram_release(df, rng):
    """Release 1: noisy model-1 histogram, then model 1 and the group means with naive and noise-aware SEs; returns
    (results, epsilon spent, scale).
    """
    m, scale, cats = histogram_meas(BUDGET["histogram_model1"])
    eps = m.map(D_IN)
    assert eps <= BUDGET["histogram_model1"] + 1e-12, eps
    noisy = np.array(m(model1_rows(df)), dtype=float).reshape(3, 9)
    point = fit_histogram(noisy)
    draws = fit_histogram(noisy + discrete_laplace(scale, (MC_DRAWS, 3, 9), rng))
    out = {}
    for k, (v, samp) in point.items():
        out[k] = {"value": float(v)}
        if samp is not None:
            noise_var = float(np.var(draws[k][0]))
            se_aware = float(np.sqrt(samp + noise_var))
            out[k].update(se_naive=float(np.sqrt(samp)), se_noise_aware=se_aware,
                          p_naive=p_value(v, np.sqrt(samp)), p_noise_aware=p_value(v, se_aware))
    return out, eps, scale


# ---------- release 2: models 2 and 3 from noisy sufficient statistics ----------
def scaled_columns(df):
    """Model 2/3 complete cases -> (n, len(COLS)) matrix with every entry in [0, 1]."""
    raw = [original.Y, "union_member", "past_union_member", "gender", "famincome", "age", "educ", "state_pre"]
    d = df.dropna(subset=raw)
    assert len(d) == N23, len(d)
    u = (d["union_member"] == 1).to_numpy(float)
    cols = [u, (d["past_union_member"] == 1).to_numpy(float) * (1 - u), (d["gender"] == 2).to_numpy(float)]
    cols += [np.clip((d[c].to_numpy(float) - lo) / (hi - lo), 0, 1) for c, (lo, hi) in BOUNDS.items()]
    cols.append((d["year"] == YEARS[1]).to_numpy(float))
    cols += [(d["state_pre"] == s).to_numpy(float) for s in STATES[1:]]
    cols.append(np.clip(d[original.Y].to_numpy(float), 0, 1))
    return np.column_stack(cols)


def moment_vector(Z):
    """Column sums and the PAIRS cross-products of Z, after asserting the per-row L1 bound the sensitivity assumes."""
    I, J = np.array(PAIRS).T
    for start in range(0, len(Z), 5000):  # check the per-row L1 bound the sensitivity relies on
        z = Z[start:start + 5000]
        assert (z.sum(1) + (z[:, I] * z[:, J]).sum(1)).max() <= ROW_L1 + 1e-9
    return np.concatenate([Z.sum(0), (Z.T @ Z)[I, J]])


def moments_meas(eps):
    """OpenDP vector Laplace on the moment vector. The row -> moment map is ours; its L1 sensitivity
    D_MOMENTS comes from the public bounds (derivation above, checked on every row in moment_vector)."""
    space = (dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(COLS) + len(PAIRS)),
             dp.l1_distance(T=float))
    return search(lambda s: dp.m.make_laplace(*space, scale=s), D_MOMENTS, eps)


def fit_moments(v, with_year):
    """Noisy moments (..., K) -> unscaled coefficients and their plug-in sampling covariance."""
    m = len(COLS)
    A = np.zeros(v.shape[:-1] + (m + 1, m + 1))  # [1, z] cross-products
    A[..., 0, 0] = N23
    A[..., 0, 1:] = A[..., 1:, 0] = v[..., :m]
    I, J = np.array(PAIRS).T
    A[..., I + 1, J + 1] = A[..., J + 1, I + 1] = v[..., m:]
    for i, c in enumerate(COLS):
        if c in BINARY:
            A[..., i + 1, i + 1] = v[..., i]
    keep = [k for k, c in enumerate(["constant"] + COLS[:-1]) if with_year or c != "year_2012"]
    XX = A[..., keep, :][..., :, keep]
    Xy = A[..., keep, m]
    yy = A[..., m, m]
    beta = np.linalg.solve(XX, Xy[..., None])[..., 0]
    ssr = yy - 2 * (beta * Xy).sum(-1) + np.einsum("...i,...ij,...j->...", beta, XX, beta)
    sigma2 = np.maximum(ssr / (N23 - len(keep)), 1e-12)
    sst = yy - v[..., m - 1]**2 / N23
    # undo the [0, 1] scaling of famincome, age and educ: beta_orig = T @ beta_scaled
    names = [(["constant"] + COLS[:-1])[k] for k in keep]
    T = np.eye(len(keep))
    for c, (lo, hi) in BOUNDS.items():
        k = names.index(c)
        T[k, k] = 1 / (hi - lo)
        T[0, k] = -lo / (hi - lo)
    idx = [names.index(t) for t in TERMS]
    coef = (beta @ T.T)[..., idx]
    cov = T @ (sigma2[..., None, None] * np.linalg.inv(XX)) @ T.T
    return coef, np.diagonal(cov, axis1=-2, axis2=-1)[..., idx], 1 - ssr / sst


def moments_release(df, rng):
    """Release 2: noisy sufficient statistics, then models 2 and 3 with naive and noise-aware SEs; returns (results,
    epsilon spent, scale).
    """
    m, scale = moments_meas(BUDGET["moments_model23"])
    eps = m.map(D_MOMENTS)
    assert eps <= BUDGET["moments_model23"] + 1e-12, eps
    noisy = np.array(m(moment_vector(scaled_columns(df)).tolist()))
    redraw = noisy + rng.laplace(0, scale, (MC_DRAWS, len(noisy)))
    out = {}
    for name, with_year in [("model2", False), ("model3", True)]:
        coef, samp, r2 = fit_moments(noisy, with_year)
        noise_var = np.var(fit_moments(redraw, with_year)[0], axis=0)
        res = {"r2": {"value": float(r2)}}
        for k, t in enumerate(TERMS):
            # a noisy X'X need not be positive definite; then a plug-in sampling variance can be <= 0
            se_naive = float(np.sqrt(samp[k])) if samp[k] > 0 else None
            se_aware = float(np.sqrt(max(samp[k], 0) + noise_var[k]))
            res[t] = {"value": float(coef[k]), "se_naive": se_naive, "se_noise_aware": se_aware,
                      "p_naive": p_value(coef[k], se_naive) if se_naive else None,
                      "p_noise_aware": p_value(coef[k], se_aware)}
        out[name] = res
    return out, eps, scale


def release(data_dir, rng=None):
    """Run both releases on the data in data_dir and return the DP results, noise scales and epsilon spent."""
    rng = rng or np.random.default_rng()  # only for the Monte Carlo SE (post-processing), not for DP noise
    df = original.load(data_dir)
    h, eps_h, scale_h = histogram_release(df, rng)
    mom, eps_m, scale_m = moments_release(df, rng)
    spent = {"histogram_model1": float(eps_h), "moments_model23": float(eps_m)}
    assert sum(spent.values()) <= EPS_TOTAL + 1e-9, spent
    model1 = {k: h[k] for k in ["constant", "union_member", "past_union_member", "r2"]}
    means = {k: h[k] for k in ["mean_union", "mean_nonunion", "diff"]}
    return {"model1": model1, **mom, "group_means": means,
            "noise_scale": {"histogram_model1": scale_h, "moments_model23": scale_m},
            "epsilon_spent": spent, "epsilon_total": float(sum(spent.values()))}


def true_values(data_dir):
    """Non-private Table 1 coefficients, SEs, p-values and the group means, for comparison."""
    df = original.load(data_dir)
    out = {}
    for k, terms in original.MODELS.items():
        res, n, r2 = original.ols(*original.design(df, terms))
        out[f"model{k}"] = {"N": n, "r2": r2, **{t: {"value": res.loc[t, "coef"], "se": res.loc[t, "se"],
                                                     "p": res.loc[t, "p"]} for t in TERMS if t in res.index}}
    out["group_means"] = original.group_means(df)
    return json.loads(json.dumps(out, default=float))


if __name__ == "__main__":
    true = true_values(sys.argv[1])
    out = release(sys.argv[1])
    for mdl in ["model1", "model2", "model3"]:
        for t in ["union_member", "past_union_member"]:
            d = out[mdl][t]
            print(f"{mdl} {t}: true {true[mdl][t]['value']:.4f} (SE {true[mdl][t]['se']:.4f})  "
                  f"DP {d['value']:.4f} (noise-aware SE {d['se_noise_aware']:.4f}, p {d['p_noise_aware']:.2g}; "
                  f"naive SE {d['se_naive']:.4f})")
    g, tg = out["group_means"], true["group_means"]
    print(f"mean resentment union: true {tg['mean_union']:.4f} DP {g['mean_union']['value']:.4f}; non-union: true "
          f"{tg['mean_nonunion']:.4f} DP {g['mean_nonunion']['value']:.4f}; difference true {tg['diff']:.4f} "
          f"DP {g['diff']['value']:.4f} (noise-aware SE {g['diff']['se_noise_aware']:.4f})")
    print(f"epsilon spent {out['epsilon_spent']}  total {out['epsilon_total']:.4f} (budget {EPS_TOTAL}, delta 0)")
    print(json.dumps({"true": true, "dp": out}))
