---
id: diffprivlib-histogram-numeric
title: Histogram of a numeric column with tools.histogram and a public range
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: histogram
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.tools.histogram2d
  - diffprivlib.tools.histogramdd
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/histograms.py
source_license: MIT
---

## Goal
Release noisy bin counts for a numeric column with pure epsilon-DP. Works like `numpy.histogram`; Neighbours: replace-one (bounded DP, n public). diffprivlib calibrates each bin's geometric noise to one record being added or removed, so the epsilon passed is halved.

## Steps
1. Fix the bins from public knowledge: either `range=(L, U)` from the documented valid range with `bins=B` (an integer), or `bins=EDGES`, an explicit edge list written down in advance. Pass `range=(EDGES[0], EDGES[-1])` as well: 1-D `histogram` raises `PrivacyLeakWarning` whenever `range` is None, even with explicit edges.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `(dp_hist, edges) <- tools.histogram(values, epsilon=EPS/2, bins=B, range=(L, U), accountant=acc)`
   - Returns non-negative integer counts and the same bin edges NumPy would produce.
   - `density=True` normalises the noisy counts afterwards (post-processing).
4. Two or more columns jointly: `tools.histogram2d(x, y, epsilon=?, bins=?, range=[(Lx, Ux), (Ly, Uy)], ...)` or `tools.histogramdd(sample, ...)`.
5. Output: true histogram with the same bins and range, DP histogram, `acc.total()`.

## Privacy accounting
The whole histogram is one release. Under replace-one (bounded DP, n public) one changed record moves between two bins (L1 sensitivity 2), while diffprivlib's calibration covers sensitivity 1. Pass EPS/2; the guarantee is `epsilon_spent <- 2 * acc.total()[0]`.

## Pitfalls
- Leaving `range=None` → `PrivacyLeakWarning`: the range comes from the data, and the outer edges reveal the true min and max.
- A string `bins` such as "auto" picks bin widths from the data: also a leak. Use an integer or explicit edges.
- Passing the full EPS under replace-one doubles the real epsilon. The full EPS is right only for add/remove-one (n private).
- Values outside `range` are dropped silently, as in NumPy.
- Noisy counts do not sum to n; do not rescale them using the exact n unless n is public.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
