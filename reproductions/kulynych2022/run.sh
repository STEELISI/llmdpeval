#!/usr/bin/env bash
# Run the authors' experiment scripts unchanged (jupytext .py form of their notebooks) in a scratch copy.
# Their own env knob DATA_FILE_FORMAT points the results at results/<name>_repro.csv, so the paper snapshot
# they ship (results/<name>_paper_snapshot.csv) stays untouched for compare.py. NUM_SHUFFLES etc. = their defaults.
set -euo pipefail
cd "$(dirname "$0")"
./fetch.sh
rm -rf work && cp -r original work
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mamba}
# The XDC has 2 cores but BLAS/torch size their thread pools from the host (139 threads seen); cap them.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
MM=${MM:-$HOME/bin/micromamba}
export MPLBACKEND=Agg
mkdir -p outputs
start=$(date -u +%FT%TZ)
for s in ${SCRIPTS:-adult}; do
  (cd work && DATA_FILE_FORMAT="results/${s}_repro.csv" "$MM" run -p ../env/py38 python "$s.py") > "outputs/$s.stdout.txt" 2>&1
  cp "work/results/${s}_repro.csv" outputs/
done
cat > outputs/provenance.json <<JSON
{"started": "$start", "finished": "$(date -u +%FT%TZ)", "host": "$(hostname)",
 "authors_commit": "$(git -C original rev-parse HEAD)", "scripts": "${SCRIPTS:-adult}",
 "python": "$("$MM" run -p env/py38 python -V)",
 "note": "authors' code unchanged; README: sklearn is not deterministic, so match is within tolerance, not exact"}
JSON
echo "outputs in $(pwd)/outputs"
