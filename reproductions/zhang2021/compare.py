"""Compare our run of notebooks 4-7 with the result tables the authors put on Dataverse (data/*.csv).
Usage: python compare.py [PANDEMIC_DIR] [DATA_DIR]     defaults: work/Data/pandemic data
Each section is skipped (with a note) when the notebook that writes its input has not run yet.
Matching is by MedDRA ID (and DrugBank ID for pairs). The CSVs round beta/gamma/delta to 2 decimals and print
p-values either rounded or as LaTeX "< m x 10^e" with a truncated mantissa, so those get tolerances; the 2x2 counts
are integers and must match exactly."""
import math, os, pickle, re, sys
import pandas as pd
from scipy.stats import spearmanr

PAN = sys.argv[1] if len(sys.argv) > 1 else "work/Data/pandemic"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data"
POPS = {"uncondition": "all_patients", "male": "male", "female": "female",
        "young": "young ", "adult": "adults", "elderly": "elderly"}  # 'young ' (sic) is the Dataverse file name


def load(p):  # pickles written by our own run of the authors' notebooks (same trust as running them)
    p = os.path.join(PAN, p)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def csv(name):
    return pd.read_csv(os.path.join(DATA, name), skiprows=2, dtype=str)  # 2 caption lines above the header


def num(s):  # "1,481.04 [205.79,10,659.00]" -> 1481.04 ; "$< 1.6\times10^{-18}$" -> 1.6e-18 ; "0.11" -> 0.11
    s = str(s).strip()
    m = re.match(r"\$<\s*([\d.]+)\\times10\^\{(-?\d+)\}\$", s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    try:
        return float(s.split("[")[0].replace(",", "").strip())
    except ValueError:  # the CSVs print "- -" where beta is undefined
        return float("nan")


def pdiff(ours, theirs_str):
    """p-value check: LaTeX '< m x 10^e' (mantissa truncated to 2 digits) -> ours in [m, m+0.1) x 10^e;
    plain numbers are rounded by the authors -> absolute difference."""
    t = num(theirs_str)
    if "times" in str(theirs_str):
        return abs(math.log10(max(ours, 1e-300)) - math.log10(t))  # log10 gap; <0.03 = same 2-digit mantissa
    return abs(ours - t)


def report_counts(m, pairs):
    ok = all((m[a].astype(int) == m[b].astype(int)).all() for a, b in pairs)
    bad = sum(~(m[a].astype(int) == m[b].astype(int)) for a, b in pairs).astype(bool).sum()
    print(f"  2x2 counts identical: {len(m) - bad}/{len(m)}" + ("" if ok else "  <-- MISMATCH"))
    return ok


print("== 1. AE vs pandemic, all patients (notebook 4; Associations_between_adverse_events_and_the_pandemic.csv)")
v = load("SE_uncondition_2019_volcano.pk")
if v is None:
    print("  skipped: SE_uncondition_2019_volcano.pk not written yet")
else:
    t = csv("Associations_between_adverse_events_and_the_pandemic.csv")
    v = v.assign(SE=v.SE.astype(str))
    m = v.merge(t, left_on="SE", right_on="MedDRA ID", how="inner")
    print(f"  ours {len(v)} AEs, theirs {len(t)}, matched by MedDRA ID {len(m)}; "
          f"theirs not in ours {len(set(t['MedDRA ID']) - set(v.SE))}")
    report_counts(m, [("2019_A", "Occur2019"), ("2019_B", "Nonoccur2019"), ("2020_A", "Occur2020"), ("2020_B", "Nonoccur2020")])
    b = (m["2019_ROR"] - m["beta (95% CI)"].map(num)).abs()
    print(f"  beta (ROR): max |ours-theirs| = {b.max():.4f} over {b.notna().sum()} finite (CSV has 2 decimals: <= 0.005 is a match);"
          f" '- -' in CSV: {m['beta (95% CI)'].str.contains('- -').sum()}")
    pv = [pdiff(o, s) for o, s in zip(m.p_corrected, m["p-value"])]
    print(f"  Bonferroni p: max |ours-theirs| = {max(pv):.4g}; "
          f"significance verdict (p<0.05) agrees on {sum((m.p_corrected < .05) == (m['p-value'].map(num) < .05))}/{len(m)}")

print("\n== 2. Top100 AEs vs pandemic (Associations_between_adverse_events_and_the_pandemic_Top100.csv)")
if v is not None:
    t100 = csv("Associations_between_adverse_events_and_the_pandemic_Top100.csv")
    full = csv("Associations_between_adverse_events_and_the_pandemic.csv")
    name2id = dict(zip(full["Adverse event"].str.lower(), full["MedDRA ID"]))
    t100["SE"] = t100["Adverse event"].str.lower().map(name2id)
    print(f"  {t100.SE.notna().sum()}/100 Top100 names resolved to a MedDRA ID via the full table")
    sig = v[v.p_corrected < 0.05].sort_values("2019_ROR", ascending=False)
    # The CSV does not state how its 100 rows were chosen (not simply the top 100 by beta), so we check membership
    # in our significant set, the order, and the counts, not "our top 100".
    print(f"  Top100 AEs in our Bonferroni-significant set: {t100.SE.isin(sig.SE).sum()}/100")
    common = t100[t100.SE.isin(sig.SE)]
    r = {s: i for i, s in enumerate(sig.SE)}
    rho = spearmanr(range(len(common)), [r[s] for s in common.SE]).correlation
    print(f"  rank agreement (their order vs our beta order) on the {len(common)} shared AEs: Spearman rho = {rho:.4f}")
    m = common.merge(v, on="SE")
    report_counts(m, [("2019_A", "Occur2019"), ("2019_B", "Nonoccur2019"), ("2020_A", "Occur2019.1"), ("2020_B", "Nonoccur2019.1")])

print("\n== 3. Identified AEs per population (notebooks 5-7; Identified_adverse_events_in_*.csv; E and P pooled)")
for pop, f in POPS.items():
    fin, step2 = [], []
    for d in ("over", "under"):
        a, s = load(f"{pop}_2019_sig_{d}_drug_pair.pk"), load(f"results/SE_{pop}_2019_sig_{d}_step2.pk")
        if a is None or s is None:
            break
        fin += list(a.SE.astype(str).unique())
        step2.append(s.assign(EP="E" if d == "over" else "P"))
    else:
        t = csv(f"Identified_adverse_events_in_{f}.csv")
        t.columns = ["name", "SE", "beta", "p", "PAEAI", "EP"]
        s = pd.concat(step2).assign(SE=lambda x: x.SE.astype(str)).drop_duplicates("SE").set_index("SE").loc[sorted(set(fin))]
        ours, theirs = set(fin), set(t.SE)
        both = sorted(ours & theirs)
        line = (f"  {pop:11s} ours {len(ours):3d}  theirs {len(theirs):3d}  shared {len(both):3d}  "
                f"Jaccard {len(both) / max(1, len(ours | theirs)):.3f}")
        if both:
            tt = t.set_index("SE").loc[both]
            ss = s.loc[both]
            line += (f"  E/P agree {(tt.EP.str.strip() == ss.EP).sum()}/{len(both)}"
                     f"  max|beta| {(ss['2019_ROR'] - tt.beta.map(num)).abs().max():.3f}"
                     f"  max|PAEAI| {(ss.PIC - tt.PAEAI.astype(float)).abs().max():.3f}"
                     f"  max p log10-gap {max(pdiff(o, q) for o, q in zip(ss.p_corrected, tt.p)):.3f}")
            if len(both) > 2:
                line += f"  PAEAI Spearman {spearmanr(ss.PIC, tt.PAEAI.astype(float)).correlation:.3f}"
        print(line)
        if ours - theirs:
            print(f"    only ours:   {sorted(ours - theirs)}")
        if theirs - ours:
            print(f"    only theirs: {sorted(theirs - ours)}")
        continue
    print(f"  {pop:11s} skipped: notebook 7 / 5 outputs not written yet")

print("\n== 4. Drug-AE pairs vs pandemic, all patients (notebook 6 cell 7; Associations_between_drug_adverse_event_pairs_and_the_pandemic.csv)")
pp = [load(f"top_SE_drug_uncondition_2019_sig_{d}.pk") for d in ("over", "under")]
if any(x is None for x in pp):
    print("  skipped: top_SE_drug_uncondition_*.pk not written yet")
else:
    o = pd.concat(pp).assign(SE=lambda x: x.SE.astype(str), drug=lambda x: x.drug.astype(str))
    t = csv("Associations_between_drug_adverse_event_pairs_and_the_pandemic.csv")
    m = o.merge(t, left_on=["SE", "drug"], right_on=["MedDRA ID", "DrugBank ID"])
    print(f"  ours {len(o)} pairs, theirs {len(t)}, matched {len(m)}")
    if len(m):
        report_counts(m, [("2019_A", "Occur2019"), ("2019_B", "Nonoccur2019"), ("2020_A", "Occur2020"), ("2020_B", "Nonoccur2020")])
        print(f"  delta: max |ours-theirs| = {(m['2019_ROR'] - m[t.columns[8]].map(num)).abs().max():.4f}")

print("\n== 5. AE-drug associations, all patients (notebook 6 cell 9, faiss-matched controls; Associations_between_adverse_events_and_drugs.csv)")
pp = [load(f"uncondition_2019_sig_{d}_drug_pair.pk") for d in ("over", "under")]
if any(x is None for x in pp):
    print("  skipped: notebook 7 outputs not written yet")
else:
    o = pd.concat(pp).assign(SE=lambda x: x.SE.astype(str), drug=lambda x: x.drug.astype(str))
    t = csv("Associations_between_adverse_events_and_drugs.csv")
    m = o.merge(t, left_on=["SE", "drug"], right_on=["MedDRA ID", "DrugBank ID"])
    print(f"  ours {len(o)} pairs, theirs {len(t)}, matched {len(m)}")
    if len(m):
        report_counts(m, [("2020_A_y", "A"), ("2020_B_y", "B"), ("2020_C", "C"), ("2020_D", "D")])
        print(f"  gamma: max |ours-theirs| = {(m['2020_ROR'] - m[t.columns[8]].map(num)).abs().max():.4f}")
