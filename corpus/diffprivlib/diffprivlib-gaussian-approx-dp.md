---
id: diffprivlib-gaussian-approx-dp
title: Approximate (epsilon, delta)-DP with Gaussian and GaussianAnalytic mechanisms
framework: diffprivlib
framework_version: "0.6.6"
mechanism: gaussian
task_type: other
dp_notion: approx
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Gaussian
  - diffprivlib.mechanisms.GaussianAnalytic
  - diffprivlib.mechanisms.GaussianDiscrete
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/gaussian.py
source_license: MIT
---

## Goal
Release a real-valued statistic with (epsilon, delta)-DP using Gaussian noise, when the task explicitly allows a nonzero delta. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

Signature: `GaussianAnalytic(*, epsilon, delta, sensitivity, random_state=None)`

## Steps
1. Clip inputs to public bounds and determine the statistic's sensitivity SENS (for a single scalar, L1 and L2 sensitivity coincide).
2. DELTA comes from the task; it should be well below 1/n.
3. Pick the class:
   - `mech <- mechanisms.GaussianAnalytic(epsilon=?, delta=DELTA, sensitivity=SENS)`: analytic calibration (Balle and Wang, ICML 2018); accepts any epsilon > 0 and gives smaller noise.
   - `mechanisms.Gaussian(epsilon=?, delta=?, sensitivity=?)`: classic calibration; raises ValueError if epsilon > 1, and its theorem (Dwork and Roth 3.22) covers epsilon in (0, 1) only, so at epsilon = 1 use `GaussianAnalytic`.
   - `mechanisms.GaussianDiscrete(epsilon=?, delta=?, sensitivity=?)`: for integer values and an integer sensitivity.
4. `dp_value <- mech.randomise(true_value)`
5. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=DELTA_TOTAL)` (a delta ceiling is required), then `acc.spend(?, DELTA)`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced value moves the total by at most U - L, the sensitivity passed. Add/remove alternative (n private): use max(|L|, |U|) as the sensitivity. Mechanisms do not spend budget; record (epsilon, delta) yourself with `acc.spend`. `acc.total()` returns both totals. With `slack > 0` on the accountant, totals over many queries can use advanced composition (see the slack outline).

## Pitfalls
- An accountant with `delta=0` rejects any spend with delta > 0 (BudgetError). Set a delta ceiling.
- Neither epsilon nor delta can be zero for these classes.
- If the task says pure DP (delta = 0), use Laplace or Geometric instead.
- The tools functions (`mean`, `sum`, ...) are pure-DP only; they have no `delta` argument.
