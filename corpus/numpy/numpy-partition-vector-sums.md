---
id: numpy-partition-vector-sums
title: Several bounded sums (group counts, group x outcome, membership probabilities) for thousands of disjoint partitions in one Laplace release
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.add.at
  - numpy.polyfit
  - numpy.random.Generator.laplace
  - numpy.nan_to_num
  - pandas.DataFrame.to_numpy
source_url: https://doi.org/10.1561/0400000042
source_license: Dwork & Roth 2014 (ideas paraphrased)
---

## Goal
Release, for every partition in a public list (areas, sites, sub-areas; thousands), the sums of several non-negative columns: one-hot group flags (group sizes), flag x outcome (outcomes per group), and soft membership probabilities and probability x outcome (estimated groups). Every later number (rates per group, differences between two estimators, errors, per-region regressions on partition shares, coarser regions) is computed from this one release. Replace-one row (bounded DP, n public; every group and partition size private).

## Steps
1. Public: partition list (codebook, not `unique()` of the data), value columns, row L1 bound C (e.g. one-hot flags add 1, flag x outcome adds at most 1, each probability vector adds 1: C = number of such blocks), a rule for missing values (count as 0).
2. Per row: vector of its columns, NaN -> 0, clip below at 0, scale down rows whose L1 norm exceeds C. Use a writable copy (`to_numpy(dtype=float, copy=True)`).
3. Sum vectors per partition into a partitions x columns matrix (`add.at`); rows outside the list are dropped.
4. `noisy <- sums + laplace(0, 2C/EPS)` on every entry, ONE draw for the whole matrix.
5. Post-processing (free): ratio of two columns per partition with a public floor and clipping to the valid range; regions = sums of their partitions through a public map; estimators and error metrics computed from noisy columns; flag partitions whose denominator is only a few noise scales. A regression within a region on partition shares (ecological / Goodman regression: outcome rate on group share, group rate = intercept + slope) is fitted on noisy rates and shares; skip partitions whose noisy total is below a public floor.

## Privacy accounting
A row sits in one partition and adds at most C in L1. Replace-one removes it and adds another row, maybe elsewhere: L1 sensitivity 2C for the whole matrix (disjoint partitions, so the cost does not grow with their number). `eps_spent <- 2C/scale`, assert <= EPS. Every derived table is post-processing.

## Pitfalls
- An exact group statistic merged back onto rows (a partition mean or share attached to every row, then noised as a row value): one person changes it for the whole group, so the per-row bound fails. Only the row's own values may enter its vector.
- Noise on each final number (rate, bias, RMSE, error) with a "range" sensitivity and EPS split over hundreds of cells: valid but the noise swamps everything. Release the sums once, derive after.
- One release per partition with EPS/partitions: same problem, and slow.
- Parallel composition "charge the max" with sensitivity C: replace-one can move a row between partitions, so 2C.
- Group sizes from the data used as public denominators.
- `df.values += noise` fails on pandas 3 (read-only array); an int array cannot take float noise in place.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
