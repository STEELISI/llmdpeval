---
id: diffprivlib-quantile-percentile-median
title: Quantiles, percentiles and the median with the exponential mechanism
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.quantile
  - diffprivlib.tools.percentile
  - diffprivlib.tools.median
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/quantiles.py
source_license: MIT
---

## Goal
Release one or more quantiles (quartiles, the median) of a numeric column with pure epsilon-DP. The library uses the exponential mechanism over the gaps between sorted values (Smith, STOC 2011). Neighbouring relation: replace-one (bounded DP, n public).

Signature: `quantile(array, quant, epsilon=1.0, bounds=None, axis=None, keepdims=False, random_state=None, accountant=None, **unused_args)`

## Steps
1. Choose `bounds=(L, U)` from public knowledge of the column's range. The DP output is always a number inside these bounds.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. Pick the call that matches the task:
   - `qs <- tools.quantile(values, quant=[Q1, ..., Qk], epsilon=?, bounds=(L, U), accountant=acc)` with each Q in [0, 1];
   - `tools.percentile(values, percent=[P1, ...], ...)` with each P in [0, 100];
   - `tools.median(values, ...)` for the 0.5 quantile.
4. With a list of k quantiles, the library splits `epsilon` evenly across them and returns an array in the same order. On a capped accountant the k float shares can round above the cap; then call once per quantile with `epsilon=min(EPS/k, acc.remaining()[0])`.
5. Output: true quantiles, DP quantiles, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): the rank utility changes by at most 1 when one row is replaced (also under add/remove), so no halving. For a list of k quantiles the accountant records k spends of epsilon/k each; `acc.total()` shows the sum, which equals the `epsilon` you passed. `acc.spent_budget` lists the per-quantile entries.

## Pitfalls
- No `bounds` → `PrivacyLeakWarning`, and the bounds become the data's own min and max: not DP.
- The quantiles are released independently, so the DP outputs may come back out of order. Sorting them afterwards is post-processing and costs nothing.
- Calling `quantile` once per quantile with the full epsilon each time spends k times the budget. Pass the list in one call, or divide the budget yourself.
- Very wide bounds spread probability over empty space; at small epsilon the output can land far from the data.
- `axis=` releases one result per slice and splits epsilon across them.
- Floating-point: k equal shares of epsilon can add up to a hair above the accountant's cap (for example 9, 11, 18 or 20 shares of 1.0), and the last spend raises `BudgetError`. Give each quantile `min(SHARE, acc.remaining()[0])`; never raise the cap to make it fit.
