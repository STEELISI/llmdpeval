"""Build data/voters.parquet from the authors' FL.Anon.RData (Dataverse doi:10.7910/DVN/SVY5VF).

Terms of access: "Only academic use permitted". data/ is gitignored; never commit or publish it.
Get the RData with reproductions/imai2016/fetch.sh (md5 + sha256 checked).

Keeps the voters with self-reported race (the authors' Predict.R filter: 9,247,810 rows) and only the
columns the Table 1 / Table 2 analysis uses. Needs pyreadr (not in the repo lock), so run it in a
separate env, e.g. `micromamba create -p <env> -c conda-forge python=3.11 pyreadr pandas pyarrow`.

Usage: python build_data.py <path/to/FL.Anon.RData> <out_dir>
"""
import sys
from pathlib import Path

import pyreadr

ETH = ["whi", "bla", "his", "asi", "oth"]
SR = [f"SR.{e.upper()}" for e in ETH]
KEEP = (["District", "Precinct", "Block", "PID", "vote08"] + SR
        + [f"{g}_{e}" for g in ["p", "r_prc", "r_blk", "r_pid"] for e in ETH])


def main(rdata, out_dir):
    df = pyreadr.read_r(rdata)["fl.anon"]
    df = df[df[SR].notna().all(axis=1)][KEEP].reset_index(drop=True)
    for c in ["District", "Precinct", "Block", "PID", "vote08"] + SR:
        df[c] = df[c].astype("int32" if c in ("Precinct", "Block") else "int8")
    assert len(df) == 9_247_810, len(df)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{out_dir}/voters.parquet", index=False)
    print(f"wrote {out_dir}/voters.parquet: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
