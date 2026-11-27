---
id: diffprivlib-laplace-bounded-noise
title: Approximate DP with bounded noise (LaplaceBoundedNoise, Uniform)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: other
dp_notion: approx
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.LaplaceBoundedNoise
  - diffprivlib.mechanisms.Uniform
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/laplace.py
source_license: MIT
---

## Goal
Release a real-valued statistic with (epsilon, delta)-DP where the noise has a hard maximum size (truncated Laplacian, Geng, Ding, Guo and Kumar 2018), when the task explicitly allows a small delta and unbounded tails are unacceptable. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

Signature: `LaplaceBoundedNoise(*, epsilon, delta, sensitivity, random_state=None)`

## Steps
1. Compute the statistic from data clipped to public bounds; state its sensitivity.
2. Choose `epsilon > 0` and `0 < delta < 0.5` from the task (delta well below 1/n).
3. `mech <- mechanisms.LaplaceBoundedNoise(epsilon=?, delta=?, sensitivity=SENS)` (keyword-only; `delta` is required).
4. `dp_value <- mech.randomise(true_value)`; the noise is Laplace-shaped but cut off at a finite width that depends on epsilon, delta and sensitivity.
5. Special case epsilon = 0: `mechanisms.Uniform(delta=?, sensitivity=SENS)` adds uniform noise and gives (0, delta)-DP.
6. Record the spend: `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=DELTA_TOTAL)`, `acc.spend(epsilon, delta)`.
7. Output the true value, DP value, `acc.total()` (both epsilon and delta).

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row moves the mean by at most (U - L)/n, the sensitivity used. Add/remove alternative (n private): n is not public, so release a noisy sum and a noisy count instead. The guarantee is (epsilon, delta)-DP, not pure DP. The accountant must be created with a nonzero delta budget, or `spend(epsilon, delta)` raises `BudgetError`. Report both numbers from `acc.total()`.

## Pitfalls
- `delta=0` raises a ValueError; for pure DP use `Laplace`, `LaplaceTruncated` or `LaplaceBoundedDomain`.
- `delta >= 0.5` raises a ValueError.
- An accountant created with `delta=0` cannot record this spend.
- Bounded noise does not bound the OUTPUT range; for outputs that must stay inside [L, U] use `LaplaceBoundedDomain` (pure DP).
