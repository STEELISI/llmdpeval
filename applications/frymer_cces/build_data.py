"""Build data/cces_white.csv from the authors' replication file (not committed: the Dataverse
dataset doi:10.7910/DVN/VJUOOV states no licence).

Keeps the white respondents (race == 1) and only the columns Table 1 uses. Rows with missing
values are kept; original.py drops them per model, like R's lm.

Needs pyreadr (not in requirements-lock.txt). Usage:
  python build_data.py <path to cces_crosssectional_replication.RDS>
The file is the one fetched by reproductions/frymer2021/fetch.sh; its sha256 is checked against
reproductions/frymer2021/checksums.sha256.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pyreadr

HERE = Path(__file__).parent
SHA256 = "79cb2190db64623ebd47fd98d63983a9efeaad2698a9af7bf94f2fa920ff15b4"
COLUMNS = ["racial_resentment", "union_member", "past_union_member", "gender", "famincome", "age", "educ",
           "state_pre", "year"]


def main(rds):
    assert hashlib.sha256(Path(rds).read_bytes()).hexdigest() == SHA256, "RDS does not match the pinned checksum"
    df = next(iter(pyreadr.read_r(rds).values()))
    white = df.loc[df["race"] == 1, COLUMNS].copy()
    white["educ"] = pd.to_numeric(white["educ"])  # stored as an object column of ints in the RDS
    (HERE / "data").mkdir(exist_ok=True)
    white.to_csv(HERE / "data" / "cces_white.csv", index=False)
    print(f"wrote data/cces_white.csv: {len(white)} rows, columns {COLUMNS}")


if __name__ == "__main__":
    main(sys.argv[1])
