#!/usr/bin/env bash
# Run the authors' notebooks 3-7 unchanged, headless, in a scratch layout that matches their relative paths
# (README step 3: Code/ next to Data/{curated,pandemic/results,parsed}). Inputs are symlinked into Data/curated.
# Executed copies of the notebooks (with all printed output) go to outputs/; the originals are never touched.
# Usage: ./run.sh            # notebooks 3 4 5 6 7 in order, fresh work/
#        NOTEBOOKS="5 6 7" ./run.sh   # resume: reuses work/ (intermediate pickles from earlier notebooks)
set -euo pipefail
cd "$(dirname "$0")"
./fetch.sh
NOTEBOOKS=${NOTEBOOKS:-3 4 5 6 7}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mamba}
MM=${MM:-$HOME/bin/micromamba}
# The XDC has 2 cores but BLAS sizes its thread pool from the host; cap it.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export MPLBACKEND=Agg
if [ "$NOTEBOOKS" = "3 4 5 6 7" ] || [ ! -d work ]; then
  rm -rf work && mkdir -p work/Code work/Data/curated work/Data/pandemic/results work/Data/parsed
  cp original/[3-7]_*.ipynb work/Code/
  for f in patient_safety.pk AE_dic.pk AE_mapping.pk drug_mapping.pk; do ln -s "../../../data/$f" "work/Data/curated/$f"; done
fi
mkdir -p outputs
for n in $NOTEBOOKS; do
  nb=$(cd work/Code && ls ${n}_*.ipynb)
  echo "$(date -u +%FT%TZ) start $nb" | tee -a outputs/timing.log
  (cd work/Code && "$MM" run -p ../../env/py37 jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
     --output-dir ../../outputs --output "${nb%.ipynb}.executed" "$nb")
  echo "$(date -u +%FT%TZ) done  $nb" | tee -a outputs/timing.log
done
cat > outputs/provenance.json <<JSON
{"finished": "$(date -u +%FT%TZ)", "host": "$(hostname)", "notebooks": "$NOTEBOOKS",
 "authors_commit": "$(git -C original rev-parse HEAD)",
 "python": "$("$MM" run -p env/py37 python -V 2>&1)",
 "note": "authors' notebooks unchanged; inputs symlinked from data/ (Dataverse doi:10.7910/DVN/G9SHDA v1.0)"}
JSON
echo "outputs in $(pwd)/outputs"
