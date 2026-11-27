#!/usr/bin/env bash
# Pinned env: Python 3.9 (the notebooks' kernel was 3.9.6) + the authors' requirements.txt, exact pins.
# torch 1.9.0 is the CPU build of the same release (no GPU here). nbconvert/ipykernel only execute the notebooks.
set -euo pipefail
cd "$(dirname "$0")/.."
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mamba}
MM=${MM:-$HOME/bin/micromamba}
./fetch.sh
rm -rf env/py39 && "$MM" create -y -q -p env/py39 -c conda-forge python=3.9.6 pip
"$MM" run -p env/py39 pip install -q torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
grep -v '^torch==' original/requirements.txt > env/requirements-authors.txt
"$MM" run -p env/py39 pip install -q -r env/requirements-authors.txt nbconvert==6.1.0 ipykernel==6.2.0 jinja2==3.0.1 ipython_genutils==0.2.0
"$MM" run -p env/py39 pip check || echo "WARNING: pip check reported issues (see above)"
"$MM" run -p env/py39 pip freeze > env/pip-freeze.txt
"$MM" env export -p env/py39 --explicit > env/explicit-lock.txt
echo "env ready"
