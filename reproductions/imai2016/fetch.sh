#!/usr/bin/env bash
# Download the authors' replication package (code + data) from Harvard Dataverse, doi:10.7910/DVN/SVY5VF v1.0
# (released 2016-01-10). Terms of Access: "Only academic use permitted". The files are fetched, checksummed
# and gitignored, never committed or redistributed. Files land in original/ exactly as the authors ship them.
# The md5s Dataverse reports for all 32 files match a full download of the dataset (dataverse_files.zip).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p original
API=https://dataverse.harvard.edu/api/access/datafile
while IFS=$'\t' read -r id name; do
  [ -s "original/$name" ] || curl -fsSL -o "original/$name" "$API/$id"
done <<'LIST'
2753350	classify.R
2753345	Codebook.txt
2753333	EM1.Block.R
2753334	EM1.Precinct.R
2753330	EM2.Block.R
2753329	EM2.Precinct.R
2753323	error.R
2753342	Figure1.R
2753349	Figure2.R
2753347	Figure3.R
2753331	FL.Anon.RData
2772133	Goodman.R
2772131	KingEI.R
2753343	L2.Demo.R
2772132	Precincts.R
2753335	Predict.EM1.Block.R
2753336	Predict.EM1.Precinct.R
2753352	Predict.EM2.Block.R
2753327	Predict.EM2.Precinct.R
2753325	Predict.EM.Combine.R
2753340	Predict.L2.R
2753324	Predict.R
2772392	Replication Guide.txt
2753332	Residuals.R
2753351	Table1.R
2753326	Table2.R
2753338	Table3.R
2753344	Table4.R
2753348	Table5.R
2753346	Table6.R
2753339	Table7.R
2753341	Turnout.R
LIST
md5sum -c --quiet checksums.md5      # md5 as published by Dataverse
sha256sum -c --quiet checksums.sha256
echo "original/ verified (32 files)"
