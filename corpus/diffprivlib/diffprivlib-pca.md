---
id: diffprivlib-pca
title: Principal component analysis with models.PCA
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: pca
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.PCA
  - diffprivlib.models.StandardScaler
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/pca.py
source_license: MIT
---

## Goal
Compute the top principal components (and explained variance) of a numeric matrix with pure epsilon-DP, via a DP covariance eigendecomposition with row norms bounded by `data_norm` (Imtiaz and Sarwate, ICASSP 2016). Neighbouring relation: replace-one (bounded DP, n public).

Signature: `PCA(n_components=None, *, epsilon=1.0, data_norm=None, centered=False, bounds=None, copy=True, whiten=False, random_state=None, accountant=None, **unused_args)`

## Steps
1. Decide how the data is centred:
   - `centered=False` (default): the library computes a DP mean with half of epsilon, using per-feature `bounds=(L_vec, U_vec)`, then subtracts it;
   - `centered=True`: you centre beforehand with public constants or with `models.StandardScaler(epsilon=?, bounds=?)`, which spends its own budget.
2. `data_norm <- NORM`: a public upper bound on the L2 norm of any row after centring (from the per-feature bounds, the norm of the largest possible deviation vector). Rows above it are clipped.
3. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
4. `pca <- models.PCA(n_components=K, epsilon=?, data_norm=NORM, centered=?, bounds=(L_vec, U_vec), accountant=acc)`
5. `pca.fit(X)`; read `pca.components_`, `pca.explained_variance_ratio_`; `pca.transform(X_new)` is post-processing.
6. Compare with the non-private sklearn `PCA`; the sign of each component is arbitrary, so compare up to sign.

## Privacy accounting
Replace-one (bounded DP, n public): diffprivlib's PCA charges the eigenvalues with sensitivity 2 on the normalised covariance and samples the vectors with Bingham, which covers one replaced row; no halving. `fit` spends `epsilon` once on the accountant (the internal split for centring is included). A DP scaler before it spends separately; `acc.total()` shows the sum.

## Pitfalls
- No `data_norm` → `PrivacyLeakWarning`; the max row norm is taken from the data.
- No `bounds` with `centered=False` → `PrivacyLeakWarning`; the mean bounds come from the data.
- Standardising with a non-private scaler fitted on the private data leaks means and variances.
- Columns on very different scales make one component dominate; rescale with public constants first.
