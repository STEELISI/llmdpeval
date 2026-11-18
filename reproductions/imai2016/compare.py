"""Compare the authors' script outputs (outputs/) with the paper's tables (expected/).

The paper prints 3 decimals (AUC: 2), so a cell matches when our value rounded the same way equals the paper's.
Usage: python compare.py [outputs_dir]   -> prints per-table match counts and every mismatch; exit 1 on mismatch.
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "outputs"


def rows(path):
    with open(path, newline="") as f:
        return [r for r in csv.reader(line for line in f if not line.startswith("#"))]


def num(x):
    return float("nan") if x in ("NA", "NaN", "") else float(x)


def compare(name, ours_file, digits):
    exp, ours = rows(HERE / "expected" / f"{name}.csv"), rows(OUT / ours_file)
    cols, exp = exp[0][1:], exp[1:]
    ours = ours[1:]  # R write.csv: header row, then rowname + values
    assert len(ours) == len(exp), f"{name}: {len(ours)} rows vs {len(exp)} expected"
    n = ok = 0
    worst = 0.0
    for e, o in zip(exp, ours):
        for c, ev, ov in zip(cols, e[1:], o[1:]):
            ev, ov = float(ev), num(ov)
            n += 1
            if round(ov, digits) == round(ev, digits):
                ok += 1
            else:
                print(f"  {name} {e[0]:16s} {c:20s} paper {ev:+.{digits}f}  ours {ov:+.{digits + 3}f}")
            worst = max(worst, abs(ov - ev)) if ov == ov else float("inf")
    print(f"{name}: {ok}/{n} cells match at {digits} dp; max |ours - paper| = {worst:.4f}")
    return ok == n


results = []
for name, f, d in [("table1", "table1.csv", 3), ("figure1_auc", "figure1_auc.csv", 2), ("table2", "table2.csv", 3)]:
    if (OUT / f).exists():
        results.append(compare(name, f, d))
    else:
        print(f"{name}: {OUT / f} not produced yet")
        results.append(False)
sys.exit(0 if all(results) else 1)

