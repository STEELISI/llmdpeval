#!/usr/bin/env bash
# Build the pinned R 3.5.1 environment for this paper inside env/r351 (user-level, no sudo).
# run.sh defaults to MAMBA_ROOT_PREFIX=/data/mamba; set the same value here.
# Exact rebuild: micromamba create -p env/r351 --file env/explicit-lock.txt, then steps 2-3.
set -euo pipefail
cd "$(dirname "$0")/.."
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mamba}
MM=${MM:-$HOME/bin/micromamba}
if [ -f env/explicit-lock.txt ]; then "$MM" create -y -p env/r351 --file env/explicit-lock.txt
else "$MM" create -y -p env/r351 -f env/environment.yml
     "$MM" env export -p env/r351 --explicit > env/explicit-lock.txt; fi              # 1. conda-forge part
mkdir -p env/r351/etc/conda/activate.d && cp env/activate-gl.sh env/r351/etc/conda/activate.d/   # 2. libGL
"$MM" run -p env/r351 Rscript env/cran_packages.R                                 # 3. CRAN Archive 2016-01-20 pins
"$MM" run -p env/r351 Rscript -e 'ip <- installed.packages()[, c("Package", "Version")]; write.csv(ip, "env/r-packages.csv", row.names = FALSE)'
echo "env ready"
