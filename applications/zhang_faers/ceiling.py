"""What our reference DP release keeps of the paper's results: 20 unseeded runs, a sweep over K, a null check.

Metrics (against the ORIGINAL analysis, main table of 7,761 AEs):
  - significant AEs kept: share of the 159 originally significant AEs that are significant in the DP release;
    false_sig = AEs significant in the DP release but not in the original; verdict agreement over all 7,761;
  - Top-100 overlap with the original Top-100;
  - |log DP ROR - log ROR| by the AE's original 2019 + 2020 count (rare < 100, 100-999, common >= 1000).
Control = the same pipeline without noise (cap at K + score test), to separate capping from noise.
Null check: AE totals kept, 2020 counts redrawn under "no change", DP noise of the reference added; share of
p < 0.05 for a naive test (noise ignored) and for the noise-aware test (should be 5%).

Usage: python ceiling.py <data_dir> [N=20]          -> runs/ceiling.json, runs/ceiling.txt  (node; repo lock env)
       python ceiling.py --plot runs/ceiling.json   -> runs/ceiling.png (needs matplotlib, not in the lock)
"""
import json
import sys

import numpy as np

KS = [1, 2, 3, 5, 10]
SWEEP_RUNS = 5
BINS = {"rare (<100)": (1, 100), "mid (100-999)": (100, 1000), "common (>=1000)": (1000, np.inf)}


def stats(v):
    """Median, quartiles and range of a list of numbers."""
    v = np.asarray(v, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": med, "q1": q1, "q3": q3, "min": v.min(), "max": v.max()}


def score(true, out):
    """Compare one release with the original: significant AEs kept and added, verdict agreement, Top-100 overlap, log
    ROR error by count bin.
    """
    import original
    tab = original.published(true["table"])
    ts, ds = set(true["true_significant"]), set(out["dp_significant"])
    d = out["table"].loc[tab.index]
    total = tab.A19 + tab.A20
    ok = np.isfinite(tab.ror) & (tab.ror > 0) & np.isfinite(d.ror) & (d.ror > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = (np.log(d.ror) - np.log(tab.ror)).abs()
    by_bin = {}
    for name, (lo, hi) in BINS.items():
        b = (total >= lo) & (total < hi)
        by_bin[name] = {"n": int(b.sum()), "finite": float(ok[b].mean()),
                        "median_abs_log_err": float(err[b & ok].median()) if (b & ok).any() else None}
    return {"sig_kept": len(ts & ds) / len(ts), "sig_kept_n": len(ts & ds), "dp_sig_n": len(ds),
            "false_sig_n": len(ds - ts), "verdict_agree": float(np.mean([(a in ts) == (a in ds) for a in tab.index])),
            "top100_overlap": len(set(true["true_top100"]) & set(out["dp_top100"])), "log_ror_err": by_bin}


def null_check(true, scale_ae, scale_win, reps, rng):
    """Share of p < ALPHA when 2020 counts are redrawn with no real change: no DP, naive test on DP counts, noise-aware
    test.
    """
    import original
    from reference_dp import dlap_var, noise_aware_test
    tab = original.published(true["table"])
    n19, n20 = true["n"][2019], true["n"][2020]
    T = np.tile((tab.A19 + tab.A20).to_numpy(), reps)
    A20 = rng.hypergeometric(n20, n19, T)
    q = np.exp(-1 / scale_ae)
    dlap = lambda: rng.geometric(1 - q, T.size) - rng.geometric(1 - q, T.size)  # discrete Laplace (simulation)
    a19, a20 = np.clip(T - A20 + dlap(), 0, n19), np.clip(A20 + dlap(), 0, n20)
    out = {}
    for name, (a, b, va) in {"no DP": (T - A20, A20, 0.0), "naive test on DP counts": (a19, a20, 0.0),
                             "noise-aware test on DP counts": (a19, a20, dlap_var(scale_ae))}.items():
        p = noise_aware_test(a, b, n19, n20, va, dlap_var(scale_win) if va else 0.0)[1]
        out[name] = {"p_lt_05": float((p < 0.05).mean()), "hist": np.histogram(p, bins=20, range=(0, 1))[0].tolist(),
                     "p_lt_05_rare": float((p[T < 100] < 0.05).mean())}
    return out


def main(data_dir, n_runs=20, out="runs/ceiling"):
    import reference_dp as R
    rng = np.random.default_rng()
    true = R.true_values(data_dir)
    runs = [R.release(data_dir, R.K, rng=rng) for _ in range(n_runs)]
    scored = [score(true, r) for r in runs]
    control = score(true, R.release(data_dir, R.K, noise=False, rng=rng))
    sweep = {k: {"dp": [score(true, R.release(data_dir, k, rng=rng)) for _ in range(SWEEP_RUNS)],
                 "control": score(true, R.release(data_dir, k, noise=False, rng=rng))} for k in KS}
    keys = ["sig_kept", "sig_kept_n", "dp_sig_n", "false_sig_n", "verdict_agree", "top100_overlap"]
    summary = {key: stats([s[key] for s in scored]) for key in keys}
    summary["log_ror_err"] = {b: stats([s["log_ror_err"][b]["median_abs_log_err"] for s in scored]) for b in BINS}
    summary["log_ror_finite"] = {b: stats([s["log_ror_err"][b]["finite"] for s in scored]) for b in BINS}
    result = {"n_runs": n_runs, "k": R.K, "budget": R.BUDGET, "scale": runs[0]["scale"],
              "epsilon_total_max": max(r["epsilon_total"] for r in runs),
              "true": {"n": true["n"], "n_significant": len(true["true_significant"]), "n_table": int(
                  (true["table"].drop_from_table == 0).sum())},
              "summary": summary, "control": control, "sweep": sweep,
              "null_check": null_check(true, runs[0]["scale"]["ae_histogram"], runs[0]["scale"]["window_counts"], 20, rng),
              "runs": scored}
    result = json.loads(json.dumps(result, default=float))
    json.dump(result, open(out + ".json", "w"), indent=1)
    lines = [f"K = {R.K}, {n_runs} runs, epsilon total max {result['epsilon_total_max']:.4f}, scale {result['scale']}",
             f"original: {result['true']['n_significant']} significant of {result['true']['n_table']} main-table AEs"]
    for key in keys:
        s = summary[key]
        lines.append(f"{key:15s} median {s['median']:.3f} [IQR {s['q1']:.3f}, {s['q3']:.3f}] range {s['min']:.3f}-{s['max']:.3f}"
                     f"   control (no noise) {control[key]:.3f}")
    for b in BINS:
        s, f = summary["log_ror_err"][b], summary["log_ror_finite"][b]
        lines.append(f"|log ROR err| {b:16s} median {s['median']:.3f} [IQR {s['q1']:.3f}, {s['q3']:.3f}]  "
                     f"finite DP ROR {f['median']:.3f}  (n {scored[0]['log_ror_err'][b]['n']})")
    lines.append("K sweep, diagnostic only (run on private data, not used to choose K); median of %d DP runs, "
                 "control = no noise:" % SWEEP_RUNS)
    for k, v in sweep.items():
        lines.append(f"  K={k:2d} sig_kept {np.median([s['sig_kept'] for s in v['dp']]):.3f} (control {v['control']['sig_kept']:.3f})"
                     f"  false_sig {np.median([s['false_sig_n'] for s in v['dp']]):.0f} (control {v['control']['false_sig_n']})"
                     f"  top100 {np.median([s['top100_overlap'] for s in v['dp']]):.0f} (control {v['control']['top100_overlap']})")
    lines.append("null check (share p < 0.05, all AEs / rare AEs < 100):")
    for name, v in result["null_check"].items():
        lines.append(f"  {name:30s} {v['p_lt_05']:.3f} / {v['p_lt_05_rare']:.3f}")
    open(out + ".txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(path):
    """Plot the K sweep and the null check from a ceiling.json to ceiling.png next to it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = json.load(open(path))
    blue, orange, grey = "#2a78d6", "#eb6834", "#52514e"
    ks = [int(k) for k in r["sweep"]]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for a, key, total, title in [(ax[0, 0], "sig_kept_n", r["true"]["n_significant"], "Significant AEs kept"),
                                 (ax[0, 1], "top100_overlap", 100, "Top-100 overlap")]:
        dpv = np.array([[s[key] for s in r["sweep"][str(k)]["dp"]] for k in ks])
        a.fill_between(ks, dpv.min(1), dpv.max(1), color=blue, alpha=0.2, lw=0)
        a.plot(ks, np.median(dpv, 1), "-o", color=blue, lw=2, ms=8, label="DP release (median, range of runs)")
        a.plot(ks, [r["sweep"][str(k)]["control"][key] for k in ks], "--o", color=orange, lw=2, ms=8,
               label="no noise, same cap K (control)")
        a.axhline(total, color=grey, lw=1, ls=":", label=f"original ({total})")
        a.set_xscale("log"); a.set_xticks(ks); a.set_xticklabels(ks)
        a.axvline(r["k"], color=grey, lw=1, ls="--")
        a.set_xlabel(f"K = adverse events kept per report (dashed: K = {r['k']}, public rule)"); a.set_title(title); a.set_ylim(0, total * 1.08)
        a.legend(frameon=False, fontsize=9)
    for k in ks:  # AEs significant in the DP release but not in the original
        v = r["sweep"][str(k)]["dp"]
        ax[0, 0].annotate(f"+{np.median([s['false_sig_n'] for s in v]):.0f}",
                          (k, np.median([s["sig_kept_n"] for s in v]) - 12), ha="center", fontsize=9, color=grey)
    ax[0, 0].set_title("Significant AEs kept (+n: significant in the DP release only)")
    a = ax[1, 0]
    bins = list(r["summary"]["log_ror_err"])
    vals = [[s["log_ror_err"][b]["median_abs_log_err"] for s in r["runs"]] for b in bins]
    a.boxplot(vals, widths=0.4)
    a.set_xticks(range(1, len(bins) + 1)); a.set_xticklabels(bins)
    a.set_ylabel("median |log DP ROR - log ROR| per run"); a.set_yscale("log")
    a.set_title(f"ROR error by how often the AE is reported (K = {r['k']}, {r['n_runs']} runs)")
    a = ax[1, 1]
    centers = np.arange(20) * 0.05 + 0.025
    for (name, v), c in zip(r["null_check"].items(), [grey, orange, blue]):
        h = np.array(v["hist"], float)
        a.step(centers, h / h.sum(), where="mid", color=c, lw=2, label=f"{name}: p<0.05 in {v['p_lt_05']:.1%}")
    a.axhline(0.05, color=grey, lw=1, ls=":")
    a.set_xlabel("p-value"); a.set_ylabel("share of AEs")
    a.set_title("No real change simulated: p-values should be flat (dotted)")
    a.legend(frameon=False, fontsize=9)
    fig.suptitle("Zhang et al. FAERS, epsilon = 1: DP 2x2 counts per adverse event "
                 "(top: K sweep, diagnostic only, run on private data, not used to choose K)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path.replace(".json", ".png"), dpi=150)
    print("saved", path.replace(".json", ".png"))


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        plot(sys.argv[2])
    else:
        main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
