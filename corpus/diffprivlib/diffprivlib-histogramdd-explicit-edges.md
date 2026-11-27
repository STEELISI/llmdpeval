---
id: diffprivlib-histogramdd-explicit-edges
title: Multi-dimensional histogram with tools.histogramdd and explicit public bin edges
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: histogram
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.histogramdd
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/tests/tools/test_histogramdd.py
source_license: MIT
---

## Goal
Release noisy counts for every cell of a d-dimensional grid over three or more numeric columns (a data cube / marginal table), with pure epsilon-DP for replace-one neighbours (bounded DP, n public).

Signature (abridged): `histogramdd(sample, epsilon=1.0, bins=10, range=None, density=None, random_state=None, accountant=None, ...)`

## Steps
1. Stack the columns into `sample` of shape (n, d).
2. Write bin edges for each dimension from public knowledge (codebook categories, policy thresholds): `bins=[EDGES_1, ..., EDGES_d]`, each a sorted array. Explicit edges need no `range`; since 0.5 edges in `bins` take priority over `range`.
   - Alternative: integers `bins=[B1, ..., Bd]` together with `range=[(L1, U1), ..., (Ld, Ud)]`.
3. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
4. `(dp_cube, edges) <- tools.histogramdd(sample, epsilon=EPS/2, bins=[...], accountant=acc)`; `edges` is a list of d arrays.
5. Marginals are post-processing: `dp_cube.sum(axis=k)` gives a lower-dimensional table at no extra cost (noise adds up across summed cells).
6. Output: true cube from `numpy.histogramdd` with the same bins, DP cube, `acc.total()`.

## Privacy accounting
The whole cube is one release; diffprivlib calibrates it for add/remove (each record in one cell). Under replace-one (bounded DP, n public) a changed record moves between two cells, so pass EPS/2 and report `epsilon_spent <- 2 * acc.total()[0]`.

## Pitfalls
- Integer `bins` without a full `range` → `PrivacyLeakWarning`: edges come from the data's min and max.
- Values outside the outermost edges are dropped silently, as in NumPy.
- Cell count is the product of bin counts; it grows fast. Sparse cubes are dominated by noise.
- Passing the full EPS under replace-one gives a 2 x EPS guarantee; the full EPS fits add/remove-one only.
- `normed=` is ignored in 0.6.6 with a `DiffprivlibCompatibilityWarning` (no error, no normalisation); use `density=` (post-processing).
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
