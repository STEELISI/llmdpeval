#!/usr/bin/env bash
# Fetch public 2010 Census block counts by race for Florida (no personal data, no account, no key):
#   1. the original 2010 PL 94-171 redistricting file (table P2, same counts as SF1 P5), and
#   2. the Census Bureau's 2010 Demonstration Data: the privacy-protected microdata file (PPMF) made by the
#      2020 Disclosure Avoidance System at production settings (2021-06-08 release). The file is the whole
#      US (13.4 GB); it is streamed and only Florida (TABBLKST 12) block x race counts are kept.
# Race groups as in wru / the authors' data: his = Hispanic of any race; non-Hispanic whi = White alone,
# bla = Black alone, asi = Asian or NHPI alone, oth = American Indian/Alaska Native or Some Other Race alone;
# two or more races ("two") are kept but not used by the method.
# Usage: ./census_fetch.sh <out_dir>
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"; cd "$OUT"

PL=https://www2.census.gov/census_2010/01-Redistricting_File--PL_94-171/Florida/fl2010.pl.zip
[ -s fl2010.pl.zip ] || curl -fsS -o fl2010.pl.zip "$PL"
python3 -c "import zipfile; zipfile.ZipFile('fl2010.pl.zip').extractall(members=['flgeo2010.pl', 'fl000012010.pl'])"

DDP=https://www2.census.gov/programs-surveys/decennial/2020/program-management/data-product-planning/2010-demonstration-data-products/01-Redistricting_File--PL_94-171/2021-06-08_ppmf_Production_Settings/2021-06-08-ppmf_P.csv
if [ ! -s ddp_fl_block_long.csv ]; then
  curl -fsS "$DDP" | awk -F, '$2 == "12" {
      r = $11 + 0; h = $10 + 0
      c = (h == 2) ? "his" : (r == 1) ? "whi" : (r == 2) ? "bla" : (r == 4 || r == 5) ? "asi" : (r == 3 || r == 6) ? "oth" : "two"
      n[$3 "," $4 "," $6 "," c]++
    } END { for (k in n) print k "," n[k] }' > ddp_fl_block_long.csv.tmp
  mv ddp_fl_block_long.csv.tmp ddp_fl_block_long.csv
fi
ls -la "$PWD"
