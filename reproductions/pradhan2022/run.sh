#!/usr/bin/env bash
# Execute the authors' notebooks unchanged in a scratch copy, inside the pinned env. The executed notebooks
# go to outputs/; compare.py diffs every cell's text output against the outputs the authors saved.
set -euo pipefail
cd "$(dirname "$0")"
./fetch.sh
rm -rf work && cp -r original work
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mamba}
# The XDC has 2 cores but BLAS/torch size their thread pools from the host (139 threads seen); cap them.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
MM=${MM:-$HOME/bin/micromamba}
mkdir -p outputs
start=$(date -u +%FT%TZ)
for nb in ${NOTEBOOKS:-removal_expl update_expl}; do
  (cd work/explanations && "$MM" run -p ../../env/py39 jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=-1 --output "../../outputs/$nb.executed.ipynb" "$nb.ipynb") 2>&1 | tee -a run.log
done
cat > outputs/provenance.json <<JSON
{"started": "$start", "finished": "$(date -u +%FT%TZ)", "host": "$(hostname)",
 "authors_commit": "$(git -C original rev-parse HEAD)",
 "python": "$("$MM" run -p env/py39 python -V)",
 "note": "authors' notebooks unchanged (dataset as shipped: german); torch seeded in their code"}
JSON
echo "outputs in $(pwd)/outputs"
