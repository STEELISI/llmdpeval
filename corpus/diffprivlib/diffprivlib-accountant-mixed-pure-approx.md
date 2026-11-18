---
id: diffprivlib-accountant-mixed-pure-approx
title: One accountant for a mix of pure (Laplace) and approximate (Gaussian) releases
framework: diffprivlib
framework_version: "0.6.6"
mechanism: gaussian
task_type: composition
dp_notion: approx
needs_bounds: true
api_calls:
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.mechanisms.GaussianAnalytic
  - diffprivlib.tools.mean
  - diffprivlib.utils.BudgetError
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/accountant.py
source_license: MIT
---

## Goal
Combine releases with different privacy notions (a pure epsilon-DP tools call and an (epsilon, delta)-DP Gaussian release) under one stated total (EPS_TOTAL, DELTA_TOTAL). Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=DELTA_TOTAL)` with DELTA_TOTAL > 0 from the task (well below 1/n).
2. Pure release through tools: `tools.mean(values, epsilon=e1, bounds=(L, U), accountant=acc)` records `(e1, 0)`.
3. Approximate release through a mechanism object:
   - `acc.check(e2, d2)` first, so an over-budget plan fails before any noise is drawn.
   - `mech <- mechanisms.GaussianAnalytic(epsilon=e2, delta=d2, sensitivity=SENS)`; `dp_x <- mech.randomise(true_x)`.
   - `acc.spend(e2, d2)`.
4. Output both releases and `acc.total()`, which returns both epsilon and delta.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. With `slack=0`, the accountant adds epsilons and composes deltas as `1 - prod(1 - delta_i)` (about the sum for small deltas). Report both numbers from `acc.total()`; the overall guarantee is (epsilon, delta)-DP even though one part was pure.

## Pitfalls
- An accountant created with `delta=0` raises `BudgetError` on any spend with delta > 0.
- The default `BudgetAccountant()` has `epsilon=inf, delta=1.0`: it never refuses anything, so it is not a budget.
- `GaussianAnalytic` and `Gaussian` require `delta > 0`; classic `Gaussian` additionally raises for epsilon > 1, and its calibration is proved only for epsilon in (0, 1): at epsilon = 1 use `GaussianAnalytic`. Mechanisms never spend on the accountant themselves.
- Mixing in zCDP/RDP results from other libraries is not supported by this accountant; convert to (epsilon, delta) first and say how.
