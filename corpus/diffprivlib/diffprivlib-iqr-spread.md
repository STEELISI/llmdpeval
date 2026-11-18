---
id: diffprivlib-iqr-spread
title: Interquartile range and other quantile spreads with tools.quantile
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.quantile
  - diffprivlib.tools.percentile
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/quantiles.py
source_license: MIT
---

## Goal
Release an outlier-resistant spread measure, the interquartile range (Q3 - Q1) or another quantile difference, with pure epsilon-DP. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `quantile(array, quant, epsilon=1.0, bounds=None, axis=None, keepdims=False, random_state=None, accountant=None, **unused_args)`

## Steps
1. Public bounds `(L, U)` for the column from domain knowledge.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `(q1, q3) <- tools.quantile(values, [0.25, 0.75], epsilon=?, bounds=(L, U), accountant=acc)`
   - `quant` is a fraction in [0, 1]. With `tools.percentile` the same call takes `[25, 75]`.
   - With a list, diffprivlib runs one exponential-mechanism quantile per entry, each with `epsilon / len(quant)`, and spends `epsilon` in total.
4. Post-processing: sort the two values (independent noise can make q1 > q3), then `iqr <- q3_sorted - q1_sorted`.
5. Output: true IQR (`numpy.quantile`), DP quartiles, DP IQR, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): the rank utility of the exponential mechanism changes by at most 1 when one row is replaced, and also under add/remove, so the epsilon is the same for both. The accountant records one spend of `epsilon / len(quant)` per quantile; their sum, `acc.total()`, equals the `epsilon` passed.

## Pitfalls
- Passing 25 and 75 to `tools.quantile` raises a ValueError: quantiles must be in [0, 1]. Use `tools.percentile` for 0-100.
- Calling `tools.quantile` twice with the full epsilon each spends 2x epsilon.
- Without `bounds` → `PrivacyLeakWarning` and bounds from the data.
- Output resolution depends on how many distinct values fall between the bounds; very wide bounds give noisy quartiles.
- The two quartiles are not forced to be ordered; sort them (post-processing) before subtracting.
