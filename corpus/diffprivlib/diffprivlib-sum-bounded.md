---
id: diffprivlib-sum-bounded
title: Bounded sum with tools.sum
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
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
Release the sum of a numeric column with pure epsilon-DP. The library calibrates noise to one value moving anywhere inside public bounds (L, U). Neighbouring relation: replace-one (bounded DP, n public).

Signature: `sum(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=False, random_state=None, accountant=None, **unused_args)`

## Steps
1. Choose `bounds=(L, U)` from public knowledge: the documented valid range of the column (units, legal limits, a codebook). Never from the column's observed min/max.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `dp_sum <- tools.sum(values, epsilon=?, bounds=(L, U), accountant=acc)`
   - Values outside the bounds are clipped to them before summing.
   - Float input gets truncated Laplace noise. An integer `dtype=` gives geometric noise and an integer result.
4. Use `tools.nansum` with the same arguments if the column contains NaN.
5. Output: true sum, DP sum, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced value moves the sum by at most U - L, which is what `tools.sum` charges. Add/remove alternative (n private): the sensitivity is max(|L|, |U|), and `tools.sum` truncates its output with the array length n, so add Laplace noise with `mechanisms.Laplace` at max(|L|, |U|) instead. The function checks and spends `epsilon` on the accountant it is given (or on the default one). Confirm with `acc.total()` after the call. Sensitivity is derived by the library from `bounds`; you never set a noise scale.

## Pitfalls
- No `bounds` → `PrivacyLeakWarning`, and the bounds are read from the private data. That breaks DP; always pass bounds.
- Wide bounds mean more noise; bounds that cut real values bias the sum through clipping. Use the narrowest range that is publicly known.
- `axis=` splits epsilon evenly across the output cells.
- Building a mean from a DP sum and a DP count costs two queries; `tools.mean` is one query.
