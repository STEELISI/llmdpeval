"""Compare original.py with the paper (reproductions/imai2016/expected/) and with our R reproduction
(reproductions/imai2016/outputs/, gitignored; table2.csv there has full precision).

Usage: python check_original.py <data_dir> <repro_dir> [out=runs/check_original.txt]
"""
import sys

import numpy as np
import pandas as pd

import original

T2_R_COLS = {"goodman": ("good.uv.prc.bias", "good.uv.prc.rmse"), "name": ("pred1.prc.bias", "pred1.prc.rmse"),
             "bayes": ("pred4.prc.bias", "pred4.prc.rmse")}


def main(data_dir, repro_dir, out="runs/check_original.txt"):
    df = original.load(data_dir)
    sr = df[original.SR].to_numpy(float)
    preds = {n: original.predict(df, g, p) for n, (g, p) in original.PREDICTIONS.items()}
    t1 = pd.DataFrame({n: original.table1_column(original.classify(p), sr) for n, p in preds.items()})
    t2 = pd.DataFrame(original.table2(df, {"name": preds["name"], "bayes": preds["name_precinct_party"]})).T
    lines = [f"voters {len(df)} (paper: 9,247,810), precincts {df['Precinct'].nunique()} (8,828), "
             f"districts {df['District'].nunique()} (25)"]

    paper1 = pd.read_csv(f"{repro_dir}/expected/table1.csv", comment="#", index_col=0)
    d1 = (t1.round(3) - paper1).abs()
    lines.append(f"Table 1 vs paper (3 dp): {(d1 < 5e-4).sum().sum()} / {d1.size} cells equal, "
                 f"max abs diff {d1.max().max():.4f}")

    paper2 = pd.read_csv(f"{repro_dir}/expected/table2.csv", comment="#", index_col=0)
    cols = [c for c in paper2.columns if not c.startswith("ei_")]
    d2 = (t2[cols].round(3) - paper2[cols]).abs()
    lines.append(f"Table 2 (Goodman, name-only, Bayesian) vs paper (3 dp): {(d2 < 5e-4).sum().sum()} / {d2.size} "
                 f"cells equal, max abs diff {d2.max().max():.4f}")

    r2 = pd.read_csv(f"{repro_dir}/outputs/table2.csv", index_col=0)
    r2.index = t2.index
    diffs = []
    for name, (b, r) in T2_R_COLS.items():
        diffs += [(t2[f"{name}_bias"] - r2[b]).abs().max(), (t2[f"{name}_rmse"] - r2[r]).abs().max()]
    lines.append(f"Table 2 vs our R run (full precision, 60 cells): max abs diff {max(diffs):.2e}")
    r1 = pd.read_csv(f"{repro_dir}/outputs/table1.csv", index_col=0).to_numpy()
    lines.append(f"Table 1 vs our R run (R output is rounded to 3 dp): max abs diff "
                 f"{np.abs(t1.round(3).to_numpy() - r1).max():.4f}")
    text = "\n".join(lines)
    print(text)
    with open(out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
