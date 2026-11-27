"""Compare our run of Frymer & Grumbach (AJPS 2021) with the paper's printed tables (expected/main_tables.csv).
Usage: python compare.py OUTPUT_DIR
Tables 1, 3, 4 are OLS (deterministic): ours must equal the printed 3-decimal value up to rounding.
Table 2 is matching with unseeded bootstraps: ours must be within one printed standard error of the paper."""
import csv
import re
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
FILES = {"T1": "union_white_racialresentment_cces_new.html",
         "T3": "union_change_racialresentment_gainedlost_vsg.html",
         "T4": "union_racialresentment_cces_panel.html"}
# paper term -> stargazer row label(s) in the authors' output
TERMS = {"female": ["factor(gender)2", "female"], "income": ["famincome", "faminc_12", "income"],
         "education": ["educ", "education"], "constant": ["Constant"], "N": ["N", "Observations"],
         "r2": ["R-squared", "R2"], "gained_union": ["union_gained", "gained_union"],
         "lost_union": ["union_lost", "lost_union"], "lag_rr": ["racial_resentment_lag", "lag_rr"]}
NUM = re.compile(r"-?\d+\.?\d*")


def stargazer(path):
    """{(row label, model number): (coef, se)} from a stargazer html table."""
    t = pd.read_html(path)[0].fillna("")
    res, label = {}, None
    for _, row in t.iterrows():
        cells = [str(c) for c in row]
        if cells[0]:
            label = cells[0]
            for m, c in enumerate(cells[1:], 1):
                x = NUM.match(c.replace(",", ""))
                if x:
                    res[(label, m)] = [float(x.group()), None]
        elif label:
            for m, c in enumerate(cells[1:], 1):
                if c.startswith("(") and (label, m) in res:
                    res[(label, m)][1] = float(NUM.search(c).group())
    return res


tabs = {k: stargazer(out / f) for k, f in FILES.items() if (out / f).exists()}
rows, ok = [], True
for r in csv.reader(l for l in open(Path(__file__).parent / "expected/main_tables.csv") if not l.startswith("#")):
    if r[0] == "table":
        continue
    tab, model, term, coef, se = r[0], r[1], r[2], float(r[3]), (float(r[4]) if r[4] else None)
    if tab == "T2":
        f = out / ("matching_results_rr_cces.csv" if model == "current" else "matching_results_rr_cces_pastmember.csv")
        if not f.exists():
            rows.append((tab, model, term, coef, None, "MISSING")); ok = False; continue
        est, mse, n = pd.read_csv(f)["x"].tolist()
        ours = est if term == "estimate" else n
        good = abs(ours - coef) <= (se if term == "estimate" else 0)
    else:
        if tab not in tabs:
            rows.append((tab, model, term, coef, None, "MISSING")); ok = False; continue
        hit = next((tabs[tab][(l, int(model))] for l in TERMS.get(term, [term]) if (l, int(model)) in tabs[tab]), None)
        if hit is None:
            rows.append((tab, model, term, coef, None, "NO ROW")); ok = False; continue
        ours = hit[0]
        tol = 0 if term == "N" else 0.0011           # printed to 3 decimals; stargazer may print more digits
        good = abs(ours - coef) <= tol and (se is None or hit[1] is None or abs(hit[1] - se) <= tol)
    ok &= good
    rows.append((tab, model, term, coef, ours, "ok" if good else "DIFF"))

df = pd.DataFrame(rows, columns=["table", "model", "term", "paper", "ours", "status"])
print(df.to_string(index=False))
print(df.groupby("table").status.value_counts().unstack(fill_value=0).to_string())
print("MATCH" if ok else "MISMATCH")
