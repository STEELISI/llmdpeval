#!/usr/bin/env bash
# Fetch the authors' code + data (MIT license, Copyright 2021 Bogdan Kulynych, EPFL) at the commit used for this
# reproduction. The UCI Adult data is checked into their repo. Not committed here: fetched into original/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=69dea17e329db52da1fcdaa9a019d4eb33c8444b   # last push 2022-01-04
[ -d original ] || git clone -q https://github.com/spring-epfl/disparate-vulnerability original
git -C original checkout -q "$COMMIT"
test "$(git -C original rev-parse HEAD)" = "$COMMIT"
git -C original status --porcelain | grep -q . && { echo "original/ is modified"; exit 1; } || true
