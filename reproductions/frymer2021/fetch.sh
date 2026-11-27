#!/usr/bin/env bash
# Download the authors' replication package (code + data) from Harvard Dataverse, doi:10.7910/DVN/VJUOOV v2.
# The dataset states no license ("limited information on how it can be used"), so the files are fetched,
# checksummed and gitignored, never committed. Files land in original/ exactly as the authors ship them.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p original
API=https://dataverse.harvard.edu/api/access/datafile
# id  name   (format=original returns the uploaded file, e.g. .dta instead of Dataverse's .tab conversion)
while read -r id name; do
  [ -s "original/$name" ] || curl -fsSL -o "original/$name" "$API/$id?format=original"
done <<'LIST'
3759509 Labor_White_Racial_Politics_Code.R
3759514 Readme.txt
3826163 Codebook.pdf
3759516 cces_crosssectional_replication.RDS
3759515 cces_panel_replication.rds
3759510 anes_replication.RDS
3759517 anes_racial_policy_replication.RDS
3759511 vsg_replication.RDS
3759513 right_to_work_laws.dta
LIST
if [ -f checksums.sha256 ]; then sha256sum -c checksums.sha256; else (cd original && sha256sum *) | sed 's#  #  original/#' > checksums.sha256; echo "wrote checksums.sha256 (commit it)"; fi
