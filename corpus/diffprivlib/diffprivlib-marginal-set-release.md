---
id: diffprivlib-marginal-set-release
title: Set of 1-way and 2-way marginals under one budget with tools.histogram (input for synthetic data)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
  - numpy.ravel_multi_index
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/histograms.py
source_license: MIT
---

## Goal
Release several marginal tables (one-way counts per category, two-way cross-tabulations of column pairs) over public category lists, all under one epsilon, for replace-one neighbours (bounded DP, n public). The noisy tables are the only input to later steps: consistency post-processing, a synthetic data generator, or utility checks.

## Steps
1. Public inputs: the list of marginals as column tuples, e.g. `[(A,), (B,), (A, B)]`, and a public category list per column (codebook values, zeros included).
2. `acc <- BudgetAccountant()`; `EPS_TABLE <- EPS / M` with M = number of tables.
3. For each table: map each column to its index in the public list (drop rows with values outside it), then encode the cell as one integer: `codes <- ravel_multi_index([idx_A, idx_B], (K_A, K_B))`.
4. `(counts, _) <- tools.histogram(codes, epsilon=EPS_TABLE / 2, bins=K, range=(-0.5, K - 0.5), accountant=acc)` with K = K_A * K_B; reshape `counts` to (K_A, K_B).
5. `epsilon_spent <- 2 * acc.total()[0]`; assert it is at most EPS.
6. Post-processing, free: combine every noisy estimate of a shared one-way margin, rescale to the public n, fit each two-way table to its margins (iterative proportional fitting).
7. Synthetic rows: sample a root column from its noisy one-way table, then each child column from the noisy conditional table of its parent (a tree chosen from public knowledge, or the maximum mutual-information tree over noisy pair tables).

## Privacy accounting
`tools.histogram` is calibrated for add/remove (one cell moves by 1). Under replace-one a changed row leaves one cell and enters another, so each table gets half its share, and the guarantee is twice the accountant total. A row is in every table, so the M shares add up to EPS (sequential composition). Steps 6-7 read only the noisy tables.

## Pitfalls
- The noise is geometric truncated at 0: counts are non-negative but small cells are biased upward. Rescale to the public n after combining.
- `bins=K` without `range` takes the range from the data (`PrivacyLeakWarning`).
- `histogram2d` / `histogramdd` work for numeric axes; for categories, integer cell codes with a public range are simpler.
- Passing EPS_TABLE (not half) under replace-one doubles the real loss.
- MST, AIM, PrivBayes and private-pgm style generators are not in diffprivlib; they need other libraries. Keep to independent or tree models here.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
