---
id: diffprivlib-var-std-bounded
title: Bounded variance and standard deviation with tools.var and tools.std
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: variance
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.var
  - diffprivlib.tools.std
  - diffprivlib.tools.nanvar
  - diffprivlib.tools.nanstd
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/utils.py
source_license: MIT
---

## Goal
Release the variance or standard deviation of a numeric column with pure epsilon-DP, given public bounds on each value. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `var(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=False, random_state=None, accountant=None, **unused_args)`

## Steps
1. Choose `bounds=(L, U)` from public knowledge of the column's valid range.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `dp_var <- tools.var(values, epsilon=?, bounds=(L, U), accountant=acc)`, or `dp_std <- tools.std(...)` with the same arguments.
   - `std` is the square root of a DP variance and costs the same single epsilon. Do not call both unless both are needed.
   - Noise comes from `LaplaceBoundedDomain` on [0, (U - L)^2 / 4], so the result is never negative.
4. Use `tools.nanvar` / `tools.nanstd` when the column has NaN values.
5. Output: true value (population form, ddof=0), DP value, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): `tools.var` and `tools.std` use the replace-one sensitivity of the variance with public n. Add/remove alternative (n private): not covered by these tools; release noisy sums of x and x^2 and a noisy count instead. Each call checks, then spends, its `epsilon` on the accountant. Calling `mean` and `var` on the same column costs the sum of both epsilons; confirm with `acc.total()`.

## Pitfalls
- Missing `bounds` → `PrivacyLeakWarning`; bounds then come from the private data.
- Sensitivity grows with the square of the range, so loose bounds hurt the variance far more than the mean.
- The estimate corresponds to NumPy's default ddof=0; compare against that.
- Mean and variance released separately are two queries; budget both explicitly.
