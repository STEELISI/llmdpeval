#!/usr/bin/env bash
# Pinned env: Python 3.7 (README: "tested on Linux with Python 3.7"; numpy 1.16.5 and faiss 1.5.3 have no 3.8 wheels)
# with only the packages the notebooks import (+ nbconvert/ipykernel to run them headless). The authors'
# requirments.txt (sic) is passed as a pip constraints file, so every package it lists - direct or transitive -
# gets exactly their version. tensorflow, faiss-gpu etc. are in that file but imported by no notebook: not installed.
set -euo pipefail
cd "$(dirname "$0")/.."
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mamba}
MM=${MM:-$HOME/bin/micromamba}
./fetch.sh
# libopenblas + llvm-openmp: runtime libraries the faiss 1.5.3 wheel expects from the OS (see activate-faiss.sh)
rm -rf env/py37 && "$MM" create -y -q -p env/py37 -c conda-forge python=3.7 pip libopenblas llvm-openmp
mkdir -p env/py37/etc/conda/activate.d && cp env/activate-faiss.sh env/py37/etc/conda/activate.d/
"$MM" run -p env/py37 pip install -q -c original/requirments.txt \
  numpy pandas scipy statsmodels scikit-learn tqdm matplotlib seaborn plotly faiss \
  nbconvert nbformat ipykernel jupyter-client
"$MM" run -p env/py37 pip check || echo "WARNING: pip check reported issues (see above)"
"$MM" run -p env/py37 pip freeze > env/pip-freeze.txt
"$MM" env export -p env/py37 --explicit > env/explicit-lock.txt
"$MM" run -p env/py37 python -c "import numpy, pandas, scipy, statsmodels, sklearn, faiss, plotly.express, seaborn; faiss.IndexFlatIP(2); print('imports OK', numpy.__version__, pandas.__version__)"
echo "env ready"
