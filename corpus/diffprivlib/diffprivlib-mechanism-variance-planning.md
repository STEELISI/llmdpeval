---
id: diffprivlib-mechanism-variance-planning
title: Choose epsilon and a mechanism before release using mech.variance and mech.bias
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.mechanisms.LaplaceTruncated
  - diffprivlib.mechanisms.Geometric
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/base.py
source_license: MIT
---

## Goal
Plan a release without touching the data: compare the expected error of candidate mechanisms and epsilon values from their analytic `variance()` and `bias()`, pick the smallest epsilon that meets a public accuracy target, then release once. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

## Steps
1. Write the target from the task: e.g. "noise standard deviation at most T" for a statistic with known sensitivity.
2. For a grid of candidate epsilons, build mechanisms (no data needed):
   - `mechanisms.Laplace(epsilon=e, sensitivity=SENS).variance(PLACEHOLDER)`; for Laplace this is `2 * (SENS / e)**2` and does not depend on the value.
   - `mechanisms.Geometric(epsilon=e, sensitivity=SENS_INT).variance(PLACEHOLDER)` for integer statistics.
   - `mechanisms.LaplaceTruncated(epsilon=e, sensitivity=SENS, lower=L, upper=U).variance(v)` and `.bias(v)` DO depend on the value v; evaluate them at public reference points (the bounds, the midpoint), never at the true value.
3. Pick the smallest epsilon whose `sqrt(variance)` meets the target; check it fits the total budget.
4. Now compute the true statistic and release it once with that mechanism and epsilon.
5. `acc.spend(epsilon, 0)` on a `BudgetAccountant`; report the chosen epsilon, the predicted noise std, and the DP value.

## Privacy accounting
Replace-one (bounded DP, n public): pass the replace-one sensitivity (count 1, sum U - L, mean (U - L)/n). Add/remove alternative (n private): a sum needs max(|L|, |U|), and a mean then needs a noisy count. Planning calls use no data and cost nothing. Only step 4 spends; record it with `acc.spend(epsilon, 0)` because mechanisms do not charge an accountant.

## Pitfalls
- Evaluating `bias(true_value)` or `variance(true_value)` for a value-dependent mechanism and then choosing epsilon from it makes epsilon data-dependent: a leak.
- Trying several epsilons on the real data and keeping the best-looking release spends the sum of all of them.
- `Staircase.variance()` and `Vector.variance()` raise `NotImplementedError`; estimate those empirically on synthetic values.
- Variance is per release; composed or averaged releases need their own error analysis.
