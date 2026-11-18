"""Which adverse events were reported more (or less) often during the COVID-19 pandemic?

For every adverse event (AE), compare the share of FAERS reports that list it in 2020 (March 11 - September 30)
with the share in the same window of 2019: a 2x2 table (reports with / without the AE, per window), the reporting
odds ratio (ROR) with a 95% CI, a two-sided Fisher exact test, and a Bonferroni correction over all tested AEs.
Then list the Top-100 AEs: among AEs in the published table that occur in at least 100 reports in one of the two
windows, the 100 with the smallest p-value, shown in descending order of ROR.

Usage: python original.py <data_dir>   (data_dir holds reports.csv and ae_terms.csv)
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ALPHA = 0.05
TOP_K = 100
MIN_REPORTS = 100  # Top-100 candidates: AE in at least this many reports in 2019 or in 2020


def load(data_dir):
    """Read reports.csv and ae_terms.csv from data_dir."""
    reports = pd.read_csv(f"{data_dir}/reports.csv", dtype={"aes": str}, keep_default_na=False)
    terms = pd.read_csv(f"{data_dir}/ae_terms.csv", dtype={"meddra_id": str}).set_index("meddra_id")
    return reports, terms


def counts(reports):
    """Reports per window (n) and, per AE, the number of reports in each window that list it (A)."""
    n = reports.window.value_counts()
    aes = reports.assign(ae=reports.aes.str.split(";")).explode("ae")
    aes = aes[aes.ae != ""]
    A = aes.groupby(["ae", "window"]).size().unstack(fill_value=0)
    return {2019: int(n[2019]), 2020: int(n[2020])}, A


def two_by_two(n, A, terms):
    """2x2 counts, ROR with 95% CI, Fisher p, Bonferroni p and verdict for every AE not on the drop list."""
    t = terms[terms.drop_before_test == 0].join(A.rename(columns={2019: "A19", 2020: "A20"}), how="inner")
    t["B19"], t["B20"] = n[2019] - t.A19, n[2020] - t.A20
    with np.errstate(divide="ignore", invalid="ignore"):
        t["ror"] = (t.A20 * t.B19) / (t.B20 * t.A19)  # inf when A19 = 0, 0 when A20 = 0
        se = 1.96 * np.sqrt(sum(np.where(t[c] > 0, 1 / t[c], 0) for c in ["A19", "B19", "A20", "B20"]))
        t["ci_low"], t["ci_high"] = np.exp(np.log(t.ror) - se), np.exp(np.log(t.ror) + se)
    t["p"] = [fisher_exact([[a19, b19], [a20, b20]])[1] for a19, b19, a20, b20 in t[["A19", "B19", "A20", "B20"]].values]
    m = len(t)
    t["p_bonf"] = np.minimum(t.p * m, 1.0)
    t["significant"] = t.p <= ALPHA / m
    t["direction"] = np.sign(t.A20 - t.A19)  # the paper splits over/under-reported AEs by the raw count change
    return t


def published(t):
    """Rows of the paper's main table (the AEs it shows; the tests were corrected over all of t)."""
    return t[t.drop_from_table == 0]


def top100(table):
    """The Top-100 AEs: at least MIN_REPORTS in one window, smallest p, shown by descending ROR."""
    cand = table[table[["A19", "A20"]].max(axis=1) >= MIN_REPORTS]
    return cand.sort_values(["p", "ror"], ascending=[True, False]).head(TOP_K).sort_values("ror", ascending=False)


def main(data_dir):
    reports, terms = load(data_dir)
    n, A = counts(reports)
    t = two_by_two(n, A, terms)
    tab = published(t)
    top = top100(tab)
    print(f"reports: 2019 window {n[2019]}, 2020 window {n[2020]}, total {sum(n.values())}")
    print(f"AEs tested {len(t)}, significant (Bonferroni p < {ALPHA}) {int(t.significant.sum())}")
    print(f"main table: {len(tab)} AEs, significant {int(tab.significant.sum())} "
          f"(more reported in 2020: {int((tab.significant & (tab.direction > 0)).sum())}, "
          f"less: {int((tab.significant & (tab.direction < 0)).sum())})")
    print(f"\nTop-{TOP_K} AEs (>= {MIN_REPORTS} reports in one window, smallest p, sorted by ROR):")
    for ae, r in top.iterrows():
        print(f"{ae} {r['name'][:40]:40s} 2019 {r.A19:5d}/{r.B19:6d}  2020 {r.A20:5d}/{r.B20:6d}  "
              f"ROR {r.ror:6.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]  p_bonf {r.p_bonf:.2e}")
    print(json.dumps({"n_reports": {str(k): v for k, v in n.items()}, "n_tested": len(t),
                      "n_significant": int(t.significant.sum()), "n_table": len(tab),
                      "n_table_significant": int(tab.significant.sum()), "top100": list(top.index)}))


if __name__ == "__main__":
    main(sys.argv[1])
