#!/usr/bin/env bash
# Pinned env: Python 3.8 (README) + every package at the exact version in the authors' poetry.lock.
# poetry 1.1.8 itself no longer imports against current virtualenv/packaging, so the lock is installed with
# pip --no-deps from env/requirements-lock.txt (generated from poetry.lock, all pins verbatim), then pip check.
set -euo pipefail
cd "$(dirname "$0")/.."
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mamba}
MM=${MM:-$HOME/bin/micromamba}
./fetch.sh
python3 env/lock_to_requirements.py original/poetry.lock > env/requirements-lock.txt
rm -rf env/py38 && "$MM" create -y -q -p env/py38 -c conda-forge python=3.8 pip
"$MM" run -p env/py38 pip install -q --no-deps -r env/requirements-lock.txt
"$MM" run -p env/py38 pip check || echo "WARNING: pip check reported issues (see above)"
"$MM" run -p env/py38 pip freeze > env/pip-freeze.txt
"$MM" env export -p env/py38 --explicit > env/explicit-lock.txt
echo "env ready"
