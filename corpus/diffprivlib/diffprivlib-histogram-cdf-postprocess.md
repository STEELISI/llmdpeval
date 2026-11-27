---
id: diffprivlib-histogram-cdf-postprocess
title: Empirical CDF and many approximate quantiles from one DP histogram
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
  - numpy.cumsum
  - numpy.searchsorted
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/notebooks/histograms.ipynb
source_license: MIT
---

## Goal
Release a cumulative distribution (share of rows at or below each public cut point) and read off as many approximate quantiles as needed, paying epsilon only once. Pure epsilon-DP for replace-one neighbours (bounded DP, n public).

## Steps
1. Fix a public grid: `range=(L, U)` from domain knowledge and `bins=B` (resolution of the CDF), or explicit edges.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `(dp_counts, edges) <- tools.histogram(values, epsilon=EPS/2, bins=B, range=(L, U), accountant=acc)`
4. Post-processing only from here (no more data access):
   - `cdf <- cumsum(dp_counts) / sum(dp_counts)` (counts are already non-negative, so the CDF is monotone).
   - quantile q: `i <- searchsorted(cdf, q)`; report the right edge `edges[i + 1]` (or interpolate inside the bin).
5. Output: true CDF on the same grid, DP CDF, the derived quantiles, `acc.total()`.

## Privacy accounting
Only the histogram touches the data, so it is the whole cost, however many quantiles you read off. Under replace-one (bounded DP, n public) one changed record moves between two bins, while diffprivlib calibrates for one bin: pass EPS/2 and report `epsilon_spent <- 2 * acc.total()[0]`. Under add/remove pass EPS.

Compare: `tools.quantile(values, [q1, ..., qk], ...)` splits epsilon evenly over the k quantiles (each gets epsilon/k). For many quantiles, the histogram route is usually more accurate; for one or two, `tools.quantile` gives finer resolution than the bin width.

## Pitfalls
- Resolution is limited by bin width; a quantile is only known up to its bin.
- Normalising by the true n instead of `sum(dp_counts)` uses private information unless n is public.
- `range=None` → `PrivacyLeakWarning`; outer edges would reveal the min and max.
- Values outside the range are dropped, which biases the CDF; choose a range that covers the domain.
