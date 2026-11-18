"""Check original.py against the printed Table 1 (reproductions/frymer2021/expected/main_tables.csv).
Every coefficient, SE, N and R2 must equal the printed 3-decimal value after rounding.

Usage: python check_original.py <data_dir>   -> runs/check_original.txt
"""
import csv
import sys
from pathlib import Path

import original

HERE = Path(__file__).parent
EXPECTED = HERE.parent.parent / "reproductions" / "frymer2021" / "expected" / "main_tables.csv"
TERM = {"female": "female", "income": "famincome", "age": "age", "education": "educ", "constant": "constant",
        "union_member": "union_member", "past_union_member": "past_union_member"}


def main(data_dir):
    df = original.load(data_dir)
    fits = {k: original.ols(*original.design(df, t)) for k, t in original.MODELS.items()}
    lines, bad = [], 0
    for r in csv.reader(l for l in open(EXPECTED) if not l.startswith("#")):
        if r[0] != "T1":
            continue
        model, term, coef, se = int(r[1]), r[2], float(r[3]), r[4]
        res, n, r2 = fits[model]
        if term == "N":
            pairs = [("N", n, coef, 0)]
        elif term == "r2":
            pairs = [("r2", r2, coef, 3)]
        else:
            row = res.loc[TERM[term]]
            pairs = [("coef", row["coef"], coef, 3), ("se", row["se"], float(se), 3)]
        for what, ours, printed, dec in pairs:
            ok = round(ours, dec) == round(printed, dec) or (dec and abs(round(ours, dec)) == 0 == printed)
            bad += not ok
            lines.append(f"model {model} {term:18s} {what:4s} ours {ours:12.5f} printed {printed:10.3f} "
                         f"{'OK' if ok else 'MISMATCH'}")
    lines.append(f"{len(lines) - bad} of {len(lines)} printed Table 1 values match; {bad} mismatches")
    text = "\n".join(lines)
    print(text)
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check_original.txt").write_text(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
