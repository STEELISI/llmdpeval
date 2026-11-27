---
id: diffprivlib-mean-bounded
title: Bounded mean with tools.mean
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.mean
  - diffprivlib.tools.nanmean
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/utils.py
source_license: MIT
---

## Goal
Release the mean of a numeric column with pure epsilon-DP. The library treats the number of rows as known and protects any one value changing within public bounds (replace-one).

Signature: `mean(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=False, random_state=None, accountant=None, **unused_args)`

## Steps
1. Pick `bounds=(L, U)` from domain knowledge (codebook range, legal or physical range). If the task states bounds, use those exactly.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `dp_mean <- tools.mean(values, epsilon=?, bounds=(L, U), accountant=acc)`
   - Values are clipped to the bounds. Noise is Laplace truncated to [L, U], so the output always lies inside the bounds.
4. Use `tools.nanmean` if NaN values must be skipped.
5. Several columns at once: pass a 2-D array with `axis=0` and per-column bounds `bounds=(L_vec, U_vec)`. Epsilon is then split evenly across the columns.
6. Output: true mean, DP mean, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row moves the mean by at most (U - L)/n, the sensitivity used. Add/remove alternative (n private): n is not public, so release a noisy sum and a noisy count instead. One call checks and spends `epsilon` on the accountant. Read `acc.total()` afterwards. The sensitivity is computed inside the library from `bounds` and the array size; do not compute a scale yourself.

## Pitfalls
- Missing `bounds` → `PrivacyLeakWarning`, and the data's own min/max are used: not DP.
- Bounds read from a summary of the data (`describe()`, min/max) are also a leak, even when passed explicitly.
- The guarantee assumes n is public. If the task treats n as private, release a DP count separately and say so.
- `random_state` fixes the noise for reproducibility only; a real release should not use a fixed seed.
