---
id: diffprivlib-kmeans-cluster-sizes
title: DP cluster centres plus DP cluster sizes (KMeans then a histogram of labels)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.KMeans
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/k_means.py
source_license: MIT
---

## Goal
Release k cluster centres AND how many rows fall in each cluster, both with DP, under one total epsilon. Neighbours: replace-one (bounded DP, n public). The sizes need a second DP release because assigning the private rows to clusters reads the data again.

## Steps
1. Public per-feature bounds `bounds=(L_vec, U_vec)` and a public number of clusters `k`.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`; split `EPS_TOTAL = eps_fit + eps_sizes`.
3. Shift `X <- clip(X, L_vec, U_vec) - L_vec`; `km <- models.KMeans(n_clusters=k, epsilon=eps_fit/2, bounds=(0, U_vec - L_vec), accountant=acc)`; `km.fit(X)`.
4. `km.cluster_centers_ + L_vec` is the first release.
5. `labels <- km.predict(X)` (data access: the labels are private).
6. `(dp_sizes, _) <- tools.histogram(labels, epsilon=eps_sizes/2, bins=k, range=(0, k), accountant=acc)`: integer-coded labels 0..k-1 with one bin per cluster.
7. Output: DP centres, DP sizes, `acc.total()`.

## Privacy accounting
Two releases, both charged to the same accountant. Under replace-one (bounded DP, n public) a changed row can move between two clusters, so the sizes histogram (calibrated by diffprivlib for one bin) is passed eps_sizes/2 and guarantees eps_sizes. KMeans itself releases per-cluster counts and sums calibrated for add/remove, so the same move costs it twice: pass eps_fit/2 (with features shifted so each lower bound is 0, which the per-cluster sum sensitivity U - L needs). Total = `2 * acc.total()[0]` = eps_fit + eps_sizes.

## Pitfalls
- `km.labels_` and `np.bincount(km.predict(X))` are exact counts of private rows; releasing them is not covered by the KMeans epsilon.
- `KMeans` picks its number of iterations from epsilon, n, k and the dimension (not from data values) and splits its epsilon across them; there is no `max_iter` to tune, and the number of clusters is public.
- Missing `bounds` → `PrivacyLeakWarning`.
- `range=(0, k)` with `bins=k` puts label j in bin j; any other range misaligns bins and clusters.
- Passing the full eps_fit to KMeans under replace-one: its guarantee is 2 x eps_fit.
- Passing the full eps_sizes to the histogram under replace-one: its guarantee is then 2 x eps_sizes.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
