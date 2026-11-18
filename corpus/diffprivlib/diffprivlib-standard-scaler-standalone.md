---
id: diffprivlib-standard-scaler-standalone
title: Release per-feature means and standard deviations with models.StandardScaler
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: variance
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.StandardScaler
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/standard_scaler.py
source_license: MIT
---

## Goal
Standardise several numeric features (z-scores) using a DP estimate of each feature's mean and standard deviation, and release those estimates (`mean_`, `scale_`, `var_`). Pure epsilon-DP. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `StandardScaler(*, epsilon=1.0, bounds=None, copy=True, with_mean=True, with_std=True, random_state=None, accountant=None)`

## Steps
1. Public per-feature bounds `bounds=(L_vec, U_vec)`.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `scaler <- models.StandardScaler(epsilon=?, bounds=(L_vec, U_vec), accountant=acc)` (keyword-only arguments).
4. `scaler.fit(X)` clips X to the bounds and estimates:
   - with `with_std=True`: half of epsilon for the means and half for the variances, each split across the features;
   - with `with_std=False`: all of epsilon for the means.
5. Read `scaler.mean_`, `scaler.var_`, `scaler.scale_`. These are the DP release.
6. `Z <- scaler.transform(X)` is post-processing of the released statistics, but the transformed rows themselves are still private data; do not publish them.
7. Output: true means and stds, DP `mean_` and `scale_`, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): the scaler's DP means and variances use the replace-one sensitivities with public n. Add/remove alternative (n private): not covered; release noisy sums and a noisy count instead. `fit` spends `epsilon` once. A second `fit` or `partial_fit` spends again. Read `acc.total()`.

## Pitfalls
- sklearn's `preprocessing.StandardScaler` (no epsilon) fitted on private data leaks the exact means and stds into any later DP model; use the diffprivlib version.
- Missing `bounds` → `PrivacyLeakWarning`.
- `fit_transform(X)` returns the scaled private rows; releasing them is not DP.
- Bounds of the scaled data (needed by a downstream DP model) follow from the public bounds and the DP statistics, not from the scaled data.
