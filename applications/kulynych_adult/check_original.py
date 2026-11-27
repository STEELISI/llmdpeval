"""Check original.py against the authors' committed results (adult_paper_snapshot.csv, model "lr").

The authors form their 200 reps as 200 random 50/50 train/test splits of the pooled complete-case
data, seeded with np.random.RandomState(i) for i in enumerate(range(200)), i.e. seed tuple (k, k).
We use the same seeds, so the splits should be the same ones (same numpy RandomState stream).

Usage: python check_original.py <data_dir> <snapshot_csv> [reps=200] [out=runs/check_original.json]
"""
import json
import sys

import numpy as np
import pandas as pd

import original


def main(data_dir, snapshot, reps=200, out="runs/check_original.json"):
    df = original.load(data_dir)
    X, y = original.design_matrix(df)
    ours = pd.DataFrame([dict(rep=k, **original.evaluate(df, X, y, np.random.RandomState((k, k))))
                         for k in range(reps)])
    snap = pd.read_csv(snapshot, index_col=0).query("model == 'lr' and rep < @reps")
    rows = {}
    for g in ["All"] + original.RACES:
        theirs = snap.query("subgroup == @g").sort_values("rep")["target_test_acc"].to_numpy()
        mine = ours[g].to_numpy()
        diff = mine - theirs
        rows[g] = {"ours_mean": mine.mean(), "theirs_mean": theirs.mean(), "ours_sd": mine.std(ddof=1),
                   "theirs_sd": theirs.std(ddof=1), "mean_diff": diff.mean(),
                   "max_abs_paired_diff": np.abs(diff).max(), "identical_reps": int((np.abs(diff) < 1e-12).sum())}
    res = json.loads(json.dumps({"reps": reps, "groups": rows,
                                 "rep0_ours": ours.iloc[0].drop("rep").to_dict()}, default=float))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(pd.DataFrame(rows).T.round(5).to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *([int(a[2])] if len(a) > 2 else []), *a[3:])
