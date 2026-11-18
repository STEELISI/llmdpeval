---
id: diffprivlib-columnwise-mean-axis
title: Per-column means of several numeric columns with axis=0 and per-column bounds
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.mean
  - diffprivlib.tools.sum
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/utils.py
source_license: MIT
---

## Goal
Release the mean of each of several numeric columns in one call, with pure epsilon-DP for the whole vector of means (bounded DP, n public). Neighbouring relation: replace-one (bounded DP, n public).

Signature: `mean(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=False, random_state=None, accountant=None, **unused_args)`

## Steps
1. Collect the columns into a 2-D float array `X` of shape (n, d).
2. Write per-column public bounds as two vectors: `bounds=(L_vec, U_vec)`, each of length d. A scalar pair `(L, U)` is broadcast to all columns.
3. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
4. `dp_means <- tools.mean(X, epsilon=?, bounds=(L_vec, U_vec), axis=0, accountant=acc)` returns a length-d array.
   - Internally the call loops over columns and gives each one `epsilon / d`; each column spends its `epsilon / d` on the accountant (d entries in `acc.spent_budget`, summing to `epsilon`).
5. The same `axis=0` pattern works for `tools.sum`, `tools.var`, `tools.std` and their nan-versions.
6. Output: true means (`X.mean(axis=0)`), DP means, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row moves the mean by at most (U - L)/n, the sensitivity used. Add/remove alternative (n private): n is not public, so release a noisy sum and a noisy count instead. The accountant records exactly the `epsilon` you passed for the whole vector; the per-column split is internal. Read `acc.total()`; do not add the columns up yourself.

## Pitfalls
- Without `axis=0` the 2-D array is flattened and you get ONE mean over all cells, not per column.
- Per-column bounds are `(lowers, uppers)`, NOT one (L, U) pair per column. A list of pairs raises `TypeError: Bounds must be specified as a tuple`. Worse, a tuple of pairs `((L1, U1), (L2, U2))` is silently read as lowers=(L1, U1), uppers=(L2, U2): wrong clipping and nonsense output with no error.
- Each column gets only `epsilon / d`, so noise per column grows with the number of columns. Drop columns you do not need.
- Bounds read from `X.min(axis=0)` / `X.max(axis=0)` are a leak.
