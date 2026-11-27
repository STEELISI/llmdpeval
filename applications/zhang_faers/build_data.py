"""Build the analysis tables in data/ from the authors' FAERS inputs (Harvard Dataverse doi:10.7910/DVN/G9SHDA,
CC0) and their notebook 4 (github.com/mims-harvard/patient-safety, commit d3a62d0, MIT).

data/reports.csv   one row per US report filed by a health professional (qualify 1-3) and received in one of the
                   two windows (2019-03-11..2019-09-30 or 2020-03-11..2020-09-30), with the report's adverse
                   events (AEs) as MedDRA preferred-term IDs joined by ';'. Selection as in notebook 3 cells 12/17.
data/ae_terms.csv  the 24,313 MedDRA preferred terms of AE_mapping.pk (id, name, system organ class), with two
                   public flags: drop_before_test (the authors' manual drop list, notebook 4 cell 7) and
                   drop_from_table (left out of the published table: see SOC_NOT_IN_TABLE below).
Both are gitignored: AE_mapping.pk carries MedDRA terms (MedDRA licence: non-commercial).

Usage: python build_data.py <fetch_dir>   (fetch_dir = where reproductions/zhang2021/fetch.sh ran: data/, original/)
The pickles are read with pandas.read_pickle only after their sha256 matches the pinned value (same files our
reproduction ran the authors' notebooks on). Needs about 15 GB RAM (the 2.3 GB pickle of all FAERS reports is loaded once).
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SHA256 = {"patient_safety.pk": "4d1bd7a564454763ef80b861aac35bff4e480911ec577ae033a5d4db5238a2eb",
          "AE_mapping.pk": "eeb9cf25fedec909026ba941748a7437898d76148917026abc63635fb0518073"}
WINDOWS = {2019: ("2019-03-10", "2019-09-31"), 2020: ("2020-03-10", "2020-09-31")}  # notebook 3: st < date < end
# The published table (Associations_between_adverse_events_and_the_pandemic.csv, 7,761 rows) leaves out these four
# system organ classes entirely, plus 8 more terms; found by comparing it with notebook 4's output (9,943 tested
# AEs). The caption only says "manually exclusion based on expertise". Bonferroni still counts all 9,943 tests.
SOC_NOT_IN_TABLE = {"Inv", "Prod", "SocCi", "Surg"}
ALSO_NOT_IN_TABLE = {"10001323", "10018386", "10020400", "10030979", "10038979", "10079848", "10083365", "10083433"}


def sha256(p):
    """SHA-256 of a file, read in 16 MB blocks."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def drop_list(notebook):
    """The two list literals of notebook 4 cell 7 (drop_SE_meddra.extend(meddra_drop_list)), parsed, not executed."""
    src = "".join("".join(c["source"]) for c in json.load(open(notebook))["cells"] if c["cell_type"] == "code")
    ids = set()
    for name in ("drop_SE_meddra", "meddra_drop_list"):
        m = re.search(rf"^{name}\s*=\s*(\[.*?\])", src, re.M | re.S)
        ids |= set(ast.literal_eval(m.group(1)))
    return ids


def main(fetch_dir):
    fetch = Path(fetch_dir)
    for f, h in SHA256.items():
        assert sha256(fetch / "data" / f) == h, f"{f} does not match the pinned checksum"
    out = HERE / "data"
    out.mkdir(exist_ok=True)

    terms = pd.read_pickle(fetch / "data" / "AE_mapping.pk")
    terms = pd.DataFrame({"meddra_id": terms.PT.astype(str), "name": terms.PT_name, "soc": terms.SOC_abbr})
    drop = drop_list(fetch / "original" / "4_disproportionality_estimation.ipynb")
    terms["drop_before_test"] = terms.meddra_id.isin(drop).astype(int)
    terms["drop_from_table"] = (terms.soc.isin(SOC_NOT_IN_TABLE) | terms.meddra_id.isin(ALSO_NOT_IN_TABLE)).astype(int)
    terms.to_csv(out / "ae_terms.csv", index=False)
    print(f"ae_terms.csv: {len(terms)} terms, drop list {len(drop)} ids ({terms.drop_before_test.sum()} in the "
          f"mapping), drop_from_table {terms.drop_from_table.sum()}")

    df = pd.read_pickle(fetch / "data" / "patient_safety.pk")
    print(f"all reports: {len(df)}; receipt_date dtype {df.receipt_date.dtype}, qualify dtype {df.qualify.dtype}")
    df = df.loc[(df.country == "US") & df.qualify.isin(["1", "2", "3"]), ["receipt_date", "SE"]]
    date = df.receipt_date.astype(str)
    parts = []
    for year, (st, end) in WINDOWS.items():
        w = df.loc[(date > st) & (date < end)]
        parts.append(pd.DataFrame({"window": year, "aes": [";".join(sorted(set(map(str, s)))) for s in w.SE]}))
        print(f"window {year}: {len(w)} reports")
    del df
    reports = pd.concat(parts, ignore_index=True)
    reports.to_csv(out / "reports.csv", index=False)
    n_aes = reports.aes.str.count(";") + (reports.aes != "")
    print(f"reports.csv: {len(reports)} rows, {(out / 'reports.csv').stat().st_size / 1e6:.1f} MB; "
          f"AEs per report: mean {n_aes.mean():.2f}, median {n_aes.median():.0f}, max {n_aes.max()}, "
          f"reports with no AE {(n_aes == 0).sum()}")


if __name__ == "__main__":
    main(sys.argv[1])
