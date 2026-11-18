---
id: diffprivlib-integer-sum-dtype
title: Integer-valued DP sum with tools.sum(dtype=int) and geometric noise
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.sum
  - diffprivlib.tools.nansum
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/utils.py
source_license: MIT
---

## Goal
Release the total of an integer-valued column (units, visits, items) as an integer, with pure epsilon-DP and discrete noise instead of floating-point Laplace noise. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `sum(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=False, random_state=None, accountant=None, **unused_args)`

## Steps
1. Public integer bounds `(L, U)` for each value (for example 0 and a documented cap).
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `dp_total <- tools.sum(values, epsilon=?, bounds=(L, U), dtype=int, accountant=acc)`
   - With an integer `dtype`, diffprivlib switches from `LaplaceTruncated` to `GeometricTruncated` (discrete Laplace) and returns an integer.
   - Values are clipped to the bounds first; the output is truncated to `[L * n, U * n]`.
4. Use `tools.nansum` with the same arguments if the column has missing values.
5. Output the true sum, DP sum, `acc.total()`.

## Privacy accounting
One call spends `epsilon` on the accountant. Sensitivity is `U - L` (replace-one, n public), computed inside the library from `bounds`. Read `acc.total()`.

## Pitfalls
- Without `dtype`, the result is a float with continuous Laplace noise even for integer inputs; rounding afterwards is allowed, but `dtype=int` gives exact discrete noise.
- Keep bounds integer when `dtype=int`.
- `U - L` is the replace-one sensitivity. Under add/remove-one it should be `max(|L|, |U|)`; with `L = 0` these agree, with `L > 0` they do not.
- Missing `bounds` → `PrivacyLeakWarning`, bounds from the data.
