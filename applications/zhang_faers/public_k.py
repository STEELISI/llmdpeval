"""Public source for K: AEs per report (authors' drop list removed) in US health-professional FAERS reports
received March 11 - September 30 of 2013-2018, i.e. reports outside the analysed windows (FDA publishes FAERS
quarterly). Prints the percentiles behind K = 6 (spec.md, Privacy setup).

Reads data/ae_terms.csv and the full FAERS pickle at /data/apps/zhang_faers/fetch/data/patient_safety.pk.
Usage: python public_k.py   > runs/public_k.txt
"""
import numpy as np
import pandas as pd

terms = pd.read_csv("data/ae_terms.csv", dtype={"meddra_id": str})
keep = set(terms.meddra_id[terms.drop_before_test == 0])
df = pd.read_pickle("/data/apps/zhang_faers/fetch/data/patient_safety.pk")
df = df.loc[(df.country == "US") & df.qualify.isin(["1", "2", "3"]), ["receipt_date", "SE"]]
d = df.receipt_date.astype(str)
for years in [range(2013, 2019), [2018]]:
    m = np.zeros(len(df), bool)
    for y in years:
        m |= ((d > f"{y}-03-10") & (d < f"{y}-09-31")).to_numpy()
    c = np.array([len(set(map(str, s)) & keep) for s in df.SE[m]])
    q = np.percentile(c, [50, 75, 90, 95, 99])
    print(f"years {list(years)}: {m.sum()} reports; AEs per report (drop list removed) mean {c.mean():.2f}, "
          f"percentiles 50/75/90/95/99 = {q.tolist()}; share with <=k AEs: " +
          ", ".join(f"k={k}: {(c <= k).mean():.3f}" for k in [1, 2, 3, 5, 10]))
