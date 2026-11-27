"""Run the reference DP release N times (unseeded) -> runs/ceiling.json.

Usage: python ceiling.py <data_dir> [N=20] [out=runs/ceiling.json]
"""
import json
import sys

import numpy as np

import original
import reference_dp


def stats(v):
    """Median, quartiles and range of a list of numbers."""
    v = np.asarray(v, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": med, "q1": q1, "q3": q3, "min": v.min(), "max": v.max()}


def main(data_dir, n_runs=20, out="runs/ceiling.json"):
    true = reference_dp.true_values(data_dir)
    runs = [reference_dp.release(data_dir) for _ in range(n_runs)]
    groups = ["All"] + original.RACES
    summary = {
        "dp_accuracy": {g: stats([r["dp_accuracy"][g] for r in runs]) for g in groups},
        "dp_model_exact_accuracy": {g: stats([r["dp_model_exact_accuracy"][g] for r in runs]) for g in groups},
        "dp_race_counts": {g: stats([r["dp_race_counts"][g] for r in runs]) for g in original.RACES},
        "abs_err_accuracy_vs_original": {g: stats([abs(r["dp_accuracy"][g] - true["accuracy"][g]) for r in runs])
                                         for g in groups},
        "abs_err_race_counts": {g: stats([abs(r["dp_race_counts"][g] - true["race_counts"][g]) for r in runs])
                                for g in original.RACES},
    }
    result = {"n_runs": n_runs, "true": true, "summary": summary,
              "epsilon_total_max": max(r["epsilon_total"] for r in runs), "runs": runs}
    result = json.loads(json.dumps(result, default=float))
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) if i == 0 else a for i, a in enumerate(sys.argv[2:])))
