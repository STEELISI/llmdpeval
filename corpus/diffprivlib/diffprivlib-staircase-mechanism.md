---
id: diffprivlib-staircase-mechanism
title: Staircase mechanism as a lower-error alternative to Laplace at larger epsilon
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Staircase
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/staircase.py
source_license: MIT
---

## Goal
Add pure epsilon-DP noise to a real-valued statistic with the staircase mechanism (Geng and Viswanath), a "geometric mixture of uniform random variables" that has lower expected error than Laplace for the same epsilon, especially when epsilon is moderate to large. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

Signature: `Staircase(*, epsilon, sensitivity, gamma=None, random_state=None)`

## Steps
1. Compute the statistic from data clipped to public bounds; state its sensitivity.
2. `mech <- mechanisms.Staircase(epsilon=?, sensitivity=SENS)`; leave `gamma=None` to use the default `1 / (1 + exp(epsilon / 2))`, which is optimal for absolute error. A custom `gamma` must be in [0, 1].
3. `dp_value <- mech.randomise(true_value)`.
4. `acc.spend(epsilon, 0)` on a `BudgetAccountant`.
5. Output the true value, DP value, `acc.total()`.
6. To compare with Laplace, run both many times and compare empirical mean absolute error: `Staircase.variance()` raises `NotImplementedError` in 0.6.6 (`bias()` returns 0).

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row moves the mean by at most (U - L)/n, the sensitivity used. Add/remove alternative (n private): n is not public, so release a noisy sum and a noisy count instead. The cost is the epsilon passed (given a correct sensitivity). Mechanisms do not charge an accountant; record `acc.spend(epsilon, 0)`.

## Pitfalls
- No output-range control: values can leave the valid range. Clamp afterwards (post-processing) if needed.
- At small epsilon the gain over Laplace is negligible; use the simpler, better-known mechanism unless the task asks.
- `Staircase(epsilon, sensitivity)` positional arguments raise TypeError; all arguments are keyword-only.
