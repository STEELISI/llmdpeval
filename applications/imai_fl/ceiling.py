"""Run the reference DP release N times (unseeded) -> runs/ceiling.json and runs/ceiling.txt.

Usage: python ceiling.py <data_dir> [N=20] [out_prefix=runs/ceiling]
"""
import json
import sys

import numpy as np
import pandas as pd

import original as og
import reference_dp as ref


def main(data_dir, n_runs=20, out="runs/ceiling"):
    n_runs = int(n_runs)
    df = og.load(data_dir)
    ex = ref.exact_stats(df)
    sr = df[og.SR].to_numpy(float)
    preds = {k: og.predict(df, g, p) for k, (g, p) in og.PREDICTIONS.items()}
    orig = {"table1": {k: og.table1_column(og.classify(p), sr) for k, p in preds.items()},
            "table2": og.table2(df, {"name": preds["name"], "bayes": preds["name_precinct_party"]})}
    # The release's post-processing without noise must give back the original analysis.
    t1_nf, t2_nf = ref.table1_from_hists(ex[0], len(df)), ref.table2_from_parts(ex[1], ex[2])
    nf_diff = max(max(abs(t1_nf[k][c] - orig["table1"][k][c]) for k in t1_nf for c in t1_nf[k]),
                  max(abs(t2_nf[r][c] - orig["table2"][r][c]) for r in t2_nf for c in t2_nf[r]))
    assert nf_diff < 1e-9, nf_diff
    runs = [ref.release(ex, len(df)) for _ in range(n_runs)]

    def summarise(table, rows, cols):
        res = {}
        for r in rows:
            for c in cols:
                t2 = table.startswith("table2")
                v = np.array([run[table][r][c] if t2 else run[table][c][r] for run in runs])
                o = orig["table2"][r][c] if t2 else orig[table][c][r]
                q1, med, q3 = np.nanpercentile(v, [25, 50, 75])
                res[f"{r}|{c}"] = {"original": o, "median": med, "q1": q1, "q3": q3, "min": np.nanmin(v),
                                   "max": np.nanmax(v), "median_abs_err": np.nanmedian(np.abs(v - o)),
                                   "n_nan": int(np.isnan(v).sum())}
        return res

    t1_rows = list(orig["table1"]["name"])
    t2_cols = list(orig["table2"]["precinct_white"])
    s1 = summarise("table1", t1_rows, list(og.PREDICTIONS))
    # Grading bar: the UNCORRECTED DP reference ("table2"). SIMEX ("table2_simex") = expert ceiling only.
    s2 = summarise("table2", list(orig["table2"]), t2_cols)
    s2_simex = summarise("table2_simex", list(orig["table2"]), t2_cols)

    # The paper's claim from Table 2: the Bayesian prediction has a smaller RMSE than Goodman's regression
    # and than the name-only prediction. Share of runs where each comparison keeps the original's direction.
    def kept(key):
        out = {}
        for r in orig["table2"]:
            for other in ["goodman", "name"]:
                o = np.sign(orig["table2"][r]["bayes_rmse"] - orig["table2"][r][f"{other}_rmse"])
                out[f"{r}|bayes_vs_{other}_rmse"] = float(np.mean(
                    [np.sign(run[key][r]["bayes_rmse"] - run[key][r][f"{other}_rmse"]) == o for run in runs]))
        return out

    keep, keep_simex = kept("table2"), kept("table2_simex")
    result = {"n_runs": n_runs, "noise_free_max_diff_vs_original": nf_diff, "table1": s1,
              "table2": s2, "rmse_order_kept_share": keep,
              "table2_expert_simex": s2_simex, "rmse_order_kept_share_expert_simex": keep_simex,
              "epsilon_total_max": max(r["epsilon_total"] for r in runs),
              "epsilon_spent": runs[0]["epsilon_spent"], "runs": runs}
    result = json.loads(json.dumps(result, default=float))
    with open(f"{out}.json", "w") as f:
        json.dump(result, f, indent=1)

    fmt = lambda s: f"{s['original']:.3f} | {s['median']:.3f} [{s['q1']:.3f}, {s['q3']:.3f}]"
    lines = [f"{n_runs} unseeded runs of reference_dp.py; epsilon total max {result['epsilon_total_max']:.6f}; "
             f"noise-free post-processing vs original: max diff {nf_diff:.1e}",
             "BAR = the uncorrected DP reference (Table 1 + district rows of Table 2 required; precinct rows reported).",
             "", "Table 1 BAR (cell: original | DP median [IQR]), prediction = name + block + party and name only:"]
    for r in t1_rows:
        lines.append(f"  {r:14s} name_block_party {fmt(s1[f'{r}|name_block_party'])}   name {fmt(s1[f'{r}|name'])}")
    for name, s, k in [("BAR, uncorrected", s2, keep),
                       ("expert ceiling, SIMEX-corrected (not the bar)", s2_simex, keep_simex)]:
        lines += ["", f"Table 2 RMSE, {name} (original | DP median [IQR]):"]
        for r in orig["table2"]:
            lines.append(f"  {r:16s} " + "  ".join(f"{c} {fmt(s[f'{r}|{c}'])}" for c in t2_cols if c.endswith("rmse")))
        lines.append("  bias, median abs error vs original: " + ", ".join(
            f"{r}/{c.split('_')[0]} {s[f'{r}|{c}']['median_abs_err']:.3f}" for r in orig["table2"] for c in t2_cols
            if c.endswith("bias")))
        lines.append(f"  share of runs keeping 'Bayesian RMSE < other RMSE' as in the original ({name}):")
        lines += [f"    {key} {v:.2f}" for key, v in k.items() if v < 1]
        lines.append(f"    all other {len([v for v in k.values() if v == 1])} comparisons: 1.00")
    text = "\n".join(lines)
    print(text)
    with open(f"{out}.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
