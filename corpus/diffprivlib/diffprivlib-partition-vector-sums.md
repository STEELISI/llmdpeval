---
id: diffprivlib-partition-vector-sums
title: Several bounded sums for thousands of disjoint partitions as one diffprivlib Laplace release with one accountant spend
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.accountant.BudgetAccountant
  - numpy.add.at
  - numpy.polyfit
  - numpy.vectorize
  - pandas.DataFrame.to_numpy
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/laplace.py
source_license: MIT
---

## Goal
Release, for every partition in a public list (areas, sites, sub-areas; thousands), the sums of several non-negative columns: one-hot group flags, flag x outcome, soft membership probabilities and probability x outcome. Rates per group, differences between estimators, error metrics, per-region regressions on partition shares and coarser regions are computed afterwards from this one release. Replace-one row (bounded DP, n public; all group and partition sizes private).

## Steps
1. Public: partition list, value columns, row L1 bound C (one per block of one-hot flags or probabilities), missing values -> 0.
2. Per row: writable float copy (`to_numpy(dtype=float, copy=True)`), NaN -> 0, clip at 0, scale rows with L1 > C down to C; sum per partition (`add.at`) into a partitions x columns matrix.
3. `acc <- BudgetAccountant()`; `mech <- Laplace(epsilon=EPS, sensitivity=2C)`; apply `mech.randomise` to every entry (`vectorize`); `acc.spend(EPS, 0)` ONCE for the matrix; noise scale from `mech.variance(0)`.
4. Post-processing: ratios with a public floor, clip to valid ranges, regions = sums of partitions, estimators and errors from noisy columns only. A regression within a region on partition shares (ecological / Goodman regression: outcome rate on group share, group rate = intercept + slope) is fitted on noisy rates and shares; skip partitions whose noisy total is below a public floor.

## Privacy accounting
One row in one partition adds at most C in L1; replace-one removes it and adds another (maybe in another partition): L1 sensitivity 2C for the whole matrix, so Laplace(2C/EPS) on each entry is one EPS-DP release. `eps_spent <- acc.total()[0]`, assert <= EPS.

## Pitfalls
- An exact group statistic merged back onto rows (a partition mean or share attached to every row, then noised as a row value): one person changes it for the whole group, so the per-row bound fails. Only the row's own values may enter its vector.
- `tools.mean` per partition (its sensitivity uses the private group size: not DP here) or `tools.sum` per partition on that partition's rows: not DP, because diffprivlib 0.6.6 truncates the output to a range set by the length of the array passed, and the partition size is private (an empty partition always returns exactly 0). A per-cell `tools.sum` is only valid on full-length masked arrays (non-members set to 0), and then very noisy.
- `tools.mean` without `bounds`: warns `PrivacyLeakWarning` and takes the bounds from the private data.
- `from diffprivlib.tools import laplace`: no such function; the mechanism is `diffprivlib.mechanisms.Laplace`.
- Noise on each final rate/bias/RMSE with a guessed sensitivity instead of releasing the sums.
- A new BudgetAccountant per query hides the total; keep one.
