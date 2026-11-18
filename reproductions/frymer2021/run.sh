#!/usr/bin/env bash
# Run the authors' R script unchanged, in a scratch copy of their package, inside the pinned R 3.5.1 env.
# Outputs (stargazer .html tables, .csv files, figures) are written by their code into work/ and copied to outputs/.
set -euo pipefail
cd "$(dirname "$0")"
./fetch.sh                                   # verifies checksums.sha256
rm -rf work && cp -r original work
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mamba}
MM=${MM:-$HOME/bin/micromamba}
start=$(date -u +%FT%TZ)
(cd work && "$MM" run -p ../env/r351 Rscript Labor_White_Racial_Politics_Code.R) 2>&1 | tee run.log
mkdir -p outputs && find work -maxdepth 1 -newer work/Readme.txt -type f -exec cp {} outputs/ \;
cat > outputs/provenance.json <<JSON
{"started": "$start", "finished": "$(date -u +%FT%TZ)", "host": "$(hostname)",
 "repo_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo none)",
 "r": "$("$MM" run -p env/r351 Rscript -e 'cat(R.version.string)')",
 "note": "authors' code unchanged; mediation/matching bootstraps are unseeded in the original, so those numbers vary run to run"}
JSON
echo "outputs in $(pwd)/outputs"
