"""Run the reference DP release N times (unseeded) -> runs/ceiling.json and runs/ceiling.txt.
Also a null calibration (no union effect) of the naive vs the noise-aware test, using the reference's own
mechanisms and scales:
  - model 1 and group means: synthetic histograms, the real group sizes, y drawn from the pooled
    distribution for every group (so the true union effect is 0), OpenDP discrete Laplace noise;
  - model 2: the real regressors with y randomly permuted across rows (so y is independent of union),
    OpenDP vector Laplace noise on the moments.
These simulations use private data internally (group sizes, regressors); nothing from them is released.

Usage: python ceiling.py <data_dir> [N=20] [null reps model 1 = 2000] [null reps model 2 = 300]
"""
import json
import sys
from pathlib import Path

import numpy as np
import opendp.prelude as dp

import original
import reference_dp as ref

HERE = Path(__file__).parent
ALPHA = 0.05


def stats(v):
    """Median, quartiles and range of a list of numbers."""
    v = np.asarray(v, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": med, "q1": q1, "q3": q3, "min": v.min(), "max": v.max()}


def summarize(runs, true):
    """Per-coefficient summary of the DP runs: values, noise-aware SEs, and how often the sign and significance are
    kept.
    """
    out = {}
    for mdl in ["model1", "model2", "model3"]:
        for t in ["union_member", "past_union_member"]:
            rs = [r[mdl][t] for r in runs]
            out[f"{mdl}.{t}"] = {
                "true": true[mdl][t]["value"], "dp": stats([r["value"] for r in rs]),
                "se_noise_aware": stats([r["se_noise_aware"] for r in rs]),
                "pct_significant_noise_aware": 100 * np.mean([r["p_noise_aware"] < ALPHA for r in rs]),
                "pct_significant_naive": 100 * np.mean([(r["p_naive"] or 1) < ALPHA for r in rs]),
                "pct_sign_kept": 100 * np.mean([np.sign(r["value"]) == np.sign(true[mdl][t]["value"]) for r in rs]),
                "pct_significant_and_sign_kept": 100 * np.mean(
                    [r["p_noise_aware"] < ALPHA and np.sign(r["value"]) == np.sign(true[mdl][t]["value"]) for r in rs]),
            }
    g, tg = [r["group_means"] for r in runs], true["group_means"]
    out["group_means"] = {
        "mean_union": {"true": tg["mean_union"], "dp": stats([x["mean_union"]["value"] for x in g])},
        "mean_nonunion": {"true": tg["mean_nonunion"], "dp": stats([x["mean_nonunion"]["value"] for x in g])},
        "diff": {"true": tg["diff"], "dp": stats([x["diff"]["value"] for x in g]),
                 "se_noise_aware": stats([x["diff"]["se_noise_aware"] for x in g]),
                 "pct_significant_noise_aware": 100 * np.mean([x["diff"]["p_noise_aware"] < ALPHA for x in g]),
                 "pct_sign_kept": 100 * np.mean([np.sign(x["diff"]["value"]) == np.sign(tg["diff"]) for x in g])},
    }
    return out


def null_histogram(df, reps, rng):
    """Model 1 union coefficient and the union / non-union difference under no union effect."""
    rows = np.array(ref.model1_rows(df))
    status = np.array([r.split("|")[0] for r in rows])
    sizes = [int((status == s).sum()) for s in ref.STATUS]
    level = np.array([int(r.split("|")[1]) for r in rows])
    probs = np.bincount(level, minlength=9) / len(level)
    _, scale, _ = ref.histogram_meas(ref.BUDGET["histogram_model1"])
    m = dp.m.make_laplace(dp.vector_domain(dp.atom_domain(T=int), size=27), dp.l1_distance(T=int), scale)
    assert m.map(ref.D_IN) <= ref.BUDGET["histogram_model1"] + 1e-12  # same mechanism as the release
    p = {k: {"naive": [], "noise_aware": []} for k in ["model1.union_member", "group_means.diff"]}
    for _ in range(reps):
        cells = np.vstack([rng.multinomial(n, probs) for n in sizes])
        noisy = np.array(m(cells.ravel().tolist()), dtype=float).reshape(3, 9)
        point = ref.fit_histogram(noisy)
        draws = ref.fit_histogram(noisy + ref.discrete_laplace(scale, (200, 3, 9), rng))
        for key, name in [("model1.union_member", "union_member"), ("group_means.diff", "diff")]:
            v, samp = point[name]
            p[key]["naive"].append(ref.p_value(v, np.sqrt(samp)))
            p[key]["noise_aware"].append(ref.p_value(v, np.sqrt(samp + np.var(draws[name][0]))))
    return p, {"group_sizes_current_past_never": sizes, "noise_scale": scale}


def null_model2(df, reps, rng):
    """Model 2 union coefficient with y permuted (independent of every regressor)."""
    Z = ref.scaled_columns(df)
    v0 = ref.moment_vector(Z)
    m_cols = len(ref.COLS)
    ypos = [m_cols + k for k, (i, j) in enumerate(ref.PAIRS) if j == m_cols - 1 and i < m_cols - 1]
    yidx = [i for (i, j) in ref.PAIRS if j == m_cols - 1 and i < m_cols - 1]
    meas, scale = ref.moments_meas(ref.BUDGET["moments_model23"])
    assert meas.map(ref.D_MOMENTS) <= ref.BUDGET["moments_model23"] + 1e-12
    k_union = ref.TERMS.index("union_member")
    p = {"naive": [], "noise_aware": [], "naive_not_computable": 0}
    for _ in range(reps):
        v = v0.copy()
        v[ypos] = Z[:, yidx].T @ rng.permutation(Z[:, -1])
        noisy = np.array(meas(v.tolist()))
        coef, samp, _ = ref.fit_moments(noisy, False)
        noise_var = np.var(ref.fit_moments(noisy + rng.laplace(0, scale, (200, len(v))), False)[0][:, k_union])
        if samp[k_union] > 0:
            p["naive"].append(ref.p_value(coef[k_union], np.sqrt(samp[k_union])))
        else:
            p["naive_not_computable"] += 1
        p["noise_aware"].append(ref.p_value(coef[k_union], np.sqrt(max(samp[k_union], 0) + noise_var)))
    return p, {"noise_scale": scale}


def noise_diagnostic(df, rng, draws=20):
    """Where model 2's noise hurts: exact moments + Laplace at the release scale on a subset of entries."""
    v = ref.moment_vector(ref.scaled_columns(df))
    _, scale = ref.moments_meas(ref.BUDGET["moments_model23"])
    names = [f"sum:{c}" for c in ref.COLS] + [f"{ref.COLS[i]}*{ref.COLS[j]}" for i, j in ref.PAIRS]
    state = np.array(["state_" in n for n in names])
    k = ref.TERMS.index("union_member")
    out = {}
    for label, mask in [("all entries", np.ones_like(state)), ("state entries only", state),
                        ("non-state entries only", ~state)]:
        c = [ref.fit_moments(v + rng.laplace(0, scale, len(v)) * mask, False)[0][k] for _ in range(draws)]
        out[label] = {"sd_union_coef": float(np.std(c)), **stats(c)}
    return out


def fpr(pvals):
    """Percentage of p-values below ALPHA (the false-positive rate under the null), or None if there are none."""
    return 100 * float(np.mean(np.array(pvals) < ALPHA)) if len(pvals) else None


def main(data_dir, n_runs=20, null_reps1=2000, null_reps2=300):
    rng = np.random.default_rng()
    df = original.load(data_dir)
    true = ref.true_values(data_dir)
    runs = [ref.release(data_dir) for _ in range(n_runs)]
    p1, info1 = null_histogram(df, null_reps1, rng)
    p2, info2 = null_model2(df, null_reps2, rng)
    null = {"alpha": ALPHA,
            "model1.union_member": {"reps": null_reps1, "fpr_naive_pct": fpr(p1["model1.union_member"]["naive"]),
                                    "fpr_noise_aware_pct": fpr(p1["model1.union_member"]["noise_aware"]), **info1},
            "group_means.diff": {"reps": null_reps1, "fpr_naive_pct": fpr(p1["group_means.diff"]["naive"]),
                                 "fpr_noise_aware_pct": fpr(p1["group_means.diff"]["noise_aware"])},
            "model2.union_member": {"reps": null_reps2, "fpr_naive_pct": fpr(p2["naive"]),
                                    "naive_not_computable": p2["naive_not_computable"],
                                    "fpr_noise_aware_pct": fpr(p2["noise_aware"]), **info2}}
    result = {"n_runs": n_runs, "true": true, "summary": summarize(runs, true), "null_calibration": null,
              "model2_noise_diagnostic": noise_diagnostic(df, rng),
              "epsilon_total_max": max(r["epsilon_total"] for r in runs), "runs": runs}
    result = json.loads(json.dumps(result, default=float))
    (HERE / "runs").mkdir(exist_ok=True)
    with open(HERE / "runs" / "ceiling.json", "w") as f:
        json.dump(result, f, indent=1)
    text = json.dumps({k: v for k, v in result.items() if k not in ("runs", "true")}, indent=1)
    (HERE / "runs" / "ceiling.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
