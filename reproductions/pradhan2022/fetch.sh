#!/usr/bin/env bash
# Fetch the authors' code + data (github.com/romilapradhan/gopher) at a pinned commit.
# The repo has NO license file, so default copyright applies: we run it for private research reproduction
# only and never commit or redistribute their code or data. It lives in original/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=e6e7eb31eca3fc7dc36323d2b02f8ba2aff0e1ce   # last push 2023-01-24
[ -d original ] || git clone -q https://github.com/romilapradhan/gopher original
git -C original checkout -q "$COMMIT"
test "$(git -C original rev-parse HEAD)" = "$COMMIT"
git -C original status --porcelain | grep -q . && { echo "original/ is modified"; exit 1; } || true
