"""Check original.py against our run of the authors' notebook 4 and against their published tables.

1. Notebook 4 output SE_uncondition_2019_volcano.pk (all tested AEs): same AEs, 2x2 counts exact, ROR, Fisher p,
   Bonferroni verdict.
2. Associations_between_adverse_events_and_the_pandemic.csv (main table): same AEs, counts exact, ROR within the
   CSV's 2-decimal rounding, verdict.
3. ..._Top100.csv: same 100 AEs, same order.

Usage: python check_original.py <data_dir> <volcano.pk> <dir with the Dataverse CSVs>  > runs/check_original.txt
The pickle is our own run of the authors' code (pinned inputs), read with pandas.read_pickle.
"""
import re
import sys

import numpy as np
import pandas as pd

import original


def num(s):
    r"""Parse a CSV cell: "13.74 [6.69,28.19]" -> 13.74, "$< 8.0\times10^{-20}$" -> 8e-20, "- -" -> nan."""
    s = str(s).strip()
    m = re.match(r"\$<\s*([\d.]+)\\times10\^\{(-?\d+)\}\$", s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    try:
        return float(s.split("[")[0].replace(",", ""))
    except ValueError:
        return float("nan")


def show(label, ok, detail=""):
    """Print one OK / FAIL check line."""
    print(f"{'OK  ' if ok else 'FAIL'} {label}{': ' + detail if detail else ''}")


def main(data_dir, volcano, csv_dir):
    reports, terms = original.load(data_dir)
    n, A = original.counts(reports)
    t = original.two_by_two(n, A, terms)
    tab = original.published(t)
    top = original.top100(tab)
    print(f"original.py: reports {n}, tested {len(t)}, main table {len(tab)}\n")

    print("== 1. vs notebook 4 output (SE_uncondition_2019_volcano.pk)")
    v = pd.read_pickle(volcano)
    v = v.assign(SE=v.SE.astype(str)).set_index("SE").drop(columns="name")
    show("same AE set", set(v.index) == set(t.index), f"ours {len(t)}, theirs {len(v)}, "
         f"only ours {len(set(t.index) - set(v.index))}, only theirs {len(set(v.index) - set(t.index))}")
    j = t.join(v, how="inner")
    bad = sum((j[a] != j[b]) for a, b in [("A19", "2019_A"), ("B19", "2019_B"), ("A20", "2020_A"), ("B20", "2020_B")])
    show("2x2 counts identical", (bad == 0).all(), f"{(bad == 0).sum()}/{len(j)}")
    fin = np.isfinite(j.ror) & np.isfinite(j["2019_ROR"])
    show("ROR", np.allclose(j.ror[fin], j["2019_ROR"][fin], rtol=1e-12) and
         ((j.ror == j["2019_ROR"]) | fin).all(), f"max relative diff {(j.ror / j['2019_ROR'] - 1)[fin].abs().max():.1e}, "
         f"non-finite agree {((j.ror == j['2019_ROR']) | fin).sum()}/{len(j)}")
    lp = (np.log10(j.p.clip(1e-300)) - np.log10(j.p_value.clip(1e-300))).abs()
    show("Fisher p", lp.max() < 1e-6, f"max |log10 ours - log10 theirs| {lp.max():.1e}")
    show("Bonferroni verdict", (j.significant == j.sig).all(), f"agree {(j.significant == j.sig).sum()}/{len(j)}, "
         f"significant ours {int(j.significant.sum())}, theirs {int(j.sig.sum())}")

    print("\n== 2. vs published main table (Associations_between_adverse_events_and_the_pandemic.csv)")
    c = pd.read_csv(f"{csv_dir}/Associations_between_adverse_events_and_the_pandemic.csv", skiprows=2,
                    dtype={"MedDRA ID": str}).set_index("MedDRA ID")
    show("same AE set", set(c.index) == set(tab.index), f"ours {len(tab)}, theirs {len(c)}")
    j = tab.join(c, how="inner")
    bad = sum((j[a] != j[b]) for a, b in [("A19", "Occur2019"), ("B19", "Nonoccur2019"), ("A20", "Occur2020"),
                                          ("B20", "Nonoccur2020")])
    show("2x2 counts identical", (bad == 0).all(), f"{(bad == 0).sum()}/{len(j)}")
    beta = j["beta (95% CI)"].map(num)
    d = (j.ror - beta).abs()[beta.notna()]
    show("ROR within CSV rounding (0.005)", d.max() <= 0.005 + 1e-9,
         f"max diff {d.max():.4f} over {len(d)} finite; CSV prints '- -' for {beta.isna().sum()} "
         f"(ours: ROR inf {np.isinf(j.ror[beta.isna()]).sum()}, 0 {(j.ror[beta.isna()] == 0).sum()})")
    pc = j["p-value"].map(num)  # "0" = below 1e-10 in this CSV
    show("verdict (p < 0.05)", ((pc < 0.05) == j.significant).all(),
         f"agree {((pc < 0.05) == j.significant).sum()}/{len(j)}, significant {int(j.significant.sum())}")

    print("\n== 3. vs published Top100 (Associations_between_adverse_events_and_the_pandemic_Top100.csv)")
    c100 = pd.read_csv(f"{csv_dir}/Associations_between_adverse_events_and_the_pandemic_Top100.csv", skiprows=2)
    name2id = dict(zip(c["Adverse event"].str.lower(), c.index))
    ids = list(c100["Adverse event"].str.lower().map(name2id))
    show("Top100 names resolved to MedDRA IDs", all(isinstance(i, str) for i in ids), f"{sum(isinstance(i, str) for i in ids)}/100")
    show("same 100 AEs", set(ids) == set(top.index), f"overlap {len(set(ids) & set(top.index))}/100")
    swaps = sum(a != b for a, b in zip(ids, top.index))
    show("same order (descending ROR, up to ties at the CSV's 2 decimals)",
         list(top.loc[ids].ror.round(2)) == list(top.ror.round(2)),
         f"{100 - swaps}/100 positions identical; the {swaps} others are neighbours whose ROR is equal at 2 decimals")
    show("Top100 counts identical", (c100[["Occur2019", "Nonoccur2019", "Occur2019.1", "Nonoccur2019.1"]].values ==
                                     top.loc[ids, ["A19", "B19", "A20", "B20"]].values).all())
    print(f"     significant in Top100: {int(top.significant.sum())}/100; 100th p_bonf {top.p_bonf.max():.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
