"""Run the reference DP release N times (unseeded) -> ceiling.json.

Usage: python ceiling.py <data_dir> [N=20] [out=ceiling.json]
"""
import json
import sys

import numpy as np

import reference_dp


def main(data_dir, n_runs=20, out="ceiling.json"):
    true = reference_dp.true_values(data_dir)
    runs = [reference_dp.release(data_dir) for _ in range(n_runs)]
    summary = {}
    for k in ["dp_accuracy", "dp_spd", "dp_model_exact_accuracy", "dp_model_exact_spd"]:
        v = np.array([r[k] for r in runs])
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        summary[k] = {"median": med, "q1": q1, "q3": q3, "min": v.min(), "max": v.max()}
    signs = [np.sign(r["dp_spd"]) == np.sign(true["spd"]) for r in runs]
    result = {"n_runs": n_runs, "true": true, "summary": summary,
              "spd_sign_preserved_pct": 100 * float(np.mean(signs)),
              "dp_accuracy_abs_err_median": float(np.median([abs(r["dp_accuracy"] - true["accuracy"]) for r in runs])),
              "dp_spd_abs_err_median": float(np.median([abs(r["dp_spd"] - true["spd"]) for r in runs])),
              "epsilon_total_max": max(r["epsilon_total"] for r in runs),
              "runs": runs}
    result = json.loads(json.dumps(result, default=float))
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) if i == 0 else a for i, a in enumerate(sys.argv[2:])))
