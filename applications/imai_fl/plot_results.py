"""Plots for runs/: census_dp.png (Census-side DP) and ceiling.png (analyst-side reference, 20 runs).
Needs matplotlib (not in the repo lock); run in a separate env.

Usage: python plot_results.py [runs_dir=runs]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACE = ["white", "black", "latino", "asian", "other"]
COLOR = dict(zip(RACE, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARK = dict(zip(RACE, ["o", "s", "^", "D", "v"]))
INK, MUTED = "#222222", "#888888"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})


def med(xs):
    """Median ignoring NaN."""
    return float(np.nanmedian(xs))


def census_plot(runs_dir):
    """Plot the Census-side results (runs_dir/census_dp.json) to census_dp.png."""
    r = json.load(open(f"{runs_dir}/census_dp.json"))
    lap, ddp, orig = r["laplace"], r["ddp"], r["original"]
    eps = sorted({x["epsilon"] for x in lap})
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))

    a = ax[0]
    for pred, col, lab in [("name_block_party", COLOR["white"], "name + block + party"),
                           ("name_precinct_party", COLOR["black"], "name + precinct + party")]:
        y = [med([x["table1"][pred]["overall_error"] for x in lap if x["epsilon"] == e]) for e in eps]
        a.plot(eps, y, color=col, lw=2, marker="o", ms=5, label=f"{lab}, discrete Laplace")
        a.axhline(orig["table1"][pred]["overall_error"], color=col, ls=":", lw=1.2)
        a.scatter([eps[-1] * 2.2], [med([x["table1"][pred]["overall_error"] for x in ddp])], color=col, marker="*",
                  s=120, zorder=5, edgecolor="white")
    a.set_xscale("log"); a.set_xlabel("epsilon of the block x race release (log)")
    a.set_ylabel("Table 1 overall error rate")
    a.set_title("Table 1: overall misclassification\n(dotted = original 2010 counts, star = 2020 DAS demo data)",
                fontsize=9, color=INK)
    a.legend(frameon=False, fontsize=8)

    a = ax[1]
    for race in RACE:
        y = [med([x["table2"][f"precinct_{race}"]["bayes_rmse"] for x in lap if x["epsilon"] == e]) for e in eps]
        a.plot(eps, y, color=COLOR[race], marker=MARK[race], ms=5, lw=2)
        a.scatter([eps[-1] * 2.2], [med([x["table2"][f"precinct_{race}"]["bayes_rmse"] for x in ddp])],
                  color=COLOR[race], marker="*", s=110, zorder=5, edgecolor="white")
        a.annotate(race, (eps[0], y[0]), xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                   fontsize=8, color=INK)
        a.axhline(orig["table2"][f"precinct_{race}"]["bayes_rmse"], color=COLOR[race], ls=":", lw=1)
    a.set_xscale("log"); a.set_xlabel("epsilon (log); star = 2020 DAS demo data")
    a.set_ylabel("RMSE of turnout by race, precinct level")
    a.set_title("Table 2, Bayesian prediction (name + precinct + party)\ndotted = original", fontsize=9, color=INK)
    a.set_xlim(eps[0] / 3, eps[-1] * 3.5)

    a = ax[2]
    for pred, col, lab in [("name_block_party", COLOR["white"], "name + block + party"),
                           ("name_precinct_party", COLOR["black"], "name + precinct + party")]:
        y = [100 * med([x["share_voters_class_changed"][pred] for x in lap if x["epsilon"] == e]) for e in eps]
        a.plot(eps, y, color=col, lw=2, marker="o", ms=5, label=lab)
        a.scatter([eps[-1] * 2.2], [100 * med([x["share_voters_class_changed"][pred] for x in ddp])], color=col,
                  marker="*", s=120, zorder=5, edgecolor="white")
    a.set_xscale("log"); a.set_xlabel("epsilon (log); star = 2020 DAS demo data")
    a.set_ylabel("% of voters whose predicted race changes")
    a.set_title("Individual predictions that flip\n(vs predictions from the original counts)", fontsize=9, color=INK)
    a.legend(frameon=False, fontsize=8)
    fig.suptitle("Census side: DP noise in the 2010 Florida block counts by race", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"{runs_dir}/census_dp.png", dpi=150)


def ceiling_plot(runs_dir):
    """Plot the analyst-side reference runs (runs_dir/ceiling.json) to ceiling.png."""
    c = json.load(open(f"{runs_dir}/ceiling.json"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    methods = [("goodman", "Goodman", -0.25), ("name", "name only", 0.0), ("bayes", "Bayesian", 0.25)]
    mcol = {"goodman": "#888888", "name": COLOR["white"], "bayes": COLOR["black"]}
    for a, level in zip(ax, ["precinct", "district"]):
        for m, lab, off in methods:
            x = np.arange(5) + off
            s = [c["table2"][f"{level}_{r}|{m}_rmse"] for r in RACE]
            a.scatter(x, [v["original"] for v in s], facecolor="white", edgecolor=mcol[m], s=40, lw=1.5, zorder=4)
            a.errorbar(x, [v["median"] for v in s],
                       yerr=[[v["median"] - v["q1"] for v in s], [v["q3"] - v["median"] for v in s]],
                       fmt="o", color=mcol[m], ms=5, capsize=0, lw=2, label=lab)
        a.set_xticks(range(5), RACE)
        a.set_title(f"{level} level: hollow = original, filled = DP median [IQR], 20 runs", fontsize=9, color=INK)
        a.set_ylabel("RMSE of turnout by race")
        a.grid(axis="x", visible=False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Analyst side: Table 2 RMSE, original vs uncorrected DP reference = the bar (epsilon = 1 total)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"{runs_dir}/ceiling.png", dpi=150)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "runs"
    census_plot(d)
    ceiling_plot(d)
