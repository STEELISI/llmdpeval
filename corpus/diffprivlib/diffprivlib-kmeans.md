---
id: diffprivlib-kmeans
title: K-means clustering with models.KMeans (DPLloyd)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.KMeans
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/k_means.py
source_license: MIT
---

## Goal
Release k cluster centres for numeric data with pure epsilon-DP, using the DPLloyd algorithm (Su, Cao, Li, Bertino and Jin, CODASPY 2016): each Lloyd iteration releases noisy per-cluster counts and sums. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `KMeans(n_clusters=8, *, epsilon=1.0, bounds=None, random_state=None, accountant=None, **unused_args)`

## Steps
1. Per-feature public bounds `bounds=(L_vec, U_vec)`. Data is clipped to them and the initial centres are drawn inside them, so the bounds also shape the result.
2. Put features on comparable public scales first (divide by documented maxima); otherwise the widest feature dominates distances.
3. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
4. Shift: `X <- clip(X, L_vec, U_vec) - L_vec`; `km <- models.KMeans(n_clusters=K, epsilon=EPS/2, bounds=(0, U_vec - L_vec), accountant=acc)`
5. `km.fit(X)`; centres = `km.cluster_centers_ + L_vec`. `km.predict(X_new)` assigns points to the released centres (post-processing).
6. Compare with the non-private sklearn `KMeans`; cluster order is arbitrary, so match centres before comparing.

## Privacy accounting
`fit` checks and spends `epsilon` once on the accountant. The library picks the number of iterations from epsilon, data size and dimensionality, and divides the budget across iterations internally. Replace-one (bounded DP, n public): each iteration releases a noisy count and sum per cluster, calibrated for add/remove, and a replaced row can move between two clusters, so the guarantee is 2 x the epsilon passed. The per-cluster sums use sensitivity U - L, which covers removing a value only when L <= 0 <= U: shift features by their public lower bounds (bounds (0, U - L)) and add L back to the centres. Pass EPS/2 and report `2 * acc.total()[0]`; assert it is <= EPS.

## Pitfalls
- No `bounds` → `PrivacyLeakWarning`; bounds are read from the data: not DP.
- `km.labels_` and `km.inertia_` are computed from the private data after fitting; publishing them is an extra, unaccounted release. Release only the centres.
- Reporting `acc.total()` under replace-one: half the real loss.
- Small epsilon or many clusters relative to n gives centres close to random points inside the bounds.
- Other sklearn KMeans arguments (`init`, `n_init`) are ignored with a warning.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
