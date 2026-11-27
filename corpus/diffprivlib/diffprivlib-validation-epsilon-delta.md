---
id: diffprivlib-validation-epsilon-delta
title: Validate a per-query (epsilon, delta) plan with validation.check_epsilon_delta before releasing
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.validation.check_epsilon_delta
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.tools.mean
  - diffprivlib.tools.var
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/tests/test_check_epsilon_delta.py
source_license: MIT
---

## Goal
Turn a budget plan given as text or config (shares per query) into checked (epsilon, delta) pairs before any data is touched, so a bad plan fails early instead of half-way through a release. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `check_epsilon_delta(epsilon, delta, allow_zero=False)`

## Steps
1. Plan: a dict query -> share, plus the total `(EPS_TOTAL, DELTA_TOTAL)`.
2. For each planned pair: `validation.check_epsilon_delta(e_i, d_i)`.
   - Raises TypeError if a value is not numeric, ValueError if epsilon < 0, if delta is outside [0, 1], or if both are zero (unless `allow_zero=True`).
3. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=DELTA_TOTAL)`; then the whole plan at once: `acc.total(spent_budget=[(e_i, d_i), ...])` must not exceed the totals. This checks the composition, which `check_epsilon_delta` does not.
4. Only then run the releases with `accountant=acc`, e.g. `tools.mean` and `tools.var` with public bounds.
5. Output the releases and `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. Validation spends nothing. The releases spend on the accountant, and `acc.total()` is the reported total.

## Pitfalls
- `check_epsilon_delta` validates ONE pair; it knows nothing about the total. Use the accountant for composition.
- Shares computed with floats may sum to slightly over 1; compare with a small tolerance or normalise the shares.
- epsilon = 0 with delta = 0 is rejected, which catches a share accidentally set to 0.
- `diffprivlib.accountant.check_epsilon_delta` is the same function re-exported; import it from `diffprivlib.validation`.
