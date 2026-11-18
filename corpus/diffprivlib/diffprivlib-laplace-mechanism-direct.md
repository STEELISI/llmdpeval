---
id: diffprivlib-laplace-mechanism-direct
title: Using the Laplace mechanism object directly, with manual accountant spend
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/laplace.py
source_license: MIT
---

## Goal
Add Laplace noise to a single real-valued statistic that no `diffprivlib.tools` function computes (a custom aggregate), when you can state its global sensitivity. Pure epsilon-DP. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

Signature: `Laplace(*, epsilon, delta=0.0, sensitivity, random_state=None)`

## Steps
1. Bound the statistic: clip inputs to public bounds (L, U) first, then aggregate to get `true_value`.
2. Work out the sensitivity of that aggregate under the task's neighbouring relation (for a clipped sum under replace-one it is U - L). State this reasoning in a comment.
3. `mech <- mechanisms.Laplace(epsilon=?, sensitivity=SENS)` (keyword-only arguments).
4. `dp_value <- mech.randomise(true_value)`. `randomise` takes one number; for arrays use a tools function.
5. `acc.spend(?, 0)` on `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`, with the same epsilon as step 3.
6. Optional diagnostics: `mech.variance(true_value)`, `mech.bias(true_value)` describe the noise.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced value moves the total by at most U - L, the sensitivity passed. Add/remove alternative (n private): use max(|L|, |U|) as the sensitivity. Mechanism objects do not touch any accountant. The privacy cost is the `epsilon` you passed, valid only if `sensitivity` is correct. Call `acc.check(epsilon, 0)` before and `acc.spend(epsilon, 0)` after, then report `acc.total()`.

## Pitfalls
- Prefer `tools.mean` / `tools.sum` / `tools.count_nonzero` when they fit: they compute sensitivity from bounds and spend on the accountant for you.
- Sensitivity measured on the actual data (local sensitivity) is not a valid input.
- Constructor arguments are keyword-only; positional arguments raise TypeError.
- Plain Laplace can return values outside the valid range (negative counts). Use `LaplaceBoundedDomain` or `LaplaceTruncated` if outputs must stay in range; clamping afterwards is also allowed (post-processing).
- Floating-point Laplace sampling has known side-channel issues (Mironov, CCS 2012); for integer counts use `Geometric`.
