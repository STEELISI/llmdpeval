---
id: numpy-partition-ratio-of-sums
title: Sums and a ratio of two sums for many disjoint partitions (small areas), with a flag where noise dominates
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.bincount
  - numpy.random.Generator.laplace
  - numpy.clip
source_url: https://doi.org/10.1561/0400000042
source_license: Dwork & Roth 2014 (ideas paraphrased)
---

## Goal
Release, for every partition in a public list (areas, sites, units; possibly thousands), the sum of two bounded non-negative columns and their ratio (e.g. a weighted rate), and say which partitions are too small for the ratio to mean anything. Replace-one neighbours (bounded DP, n public).

## Steps
1. Public inputs: the partition list (codebook), bounds `(0, U_num)` and `(0, U_den)` for the two columns, the valid range of the ratio, and fixed post-processing rules (a denominator floor, a maximum relative error).
2. Clamp both columns; per partition `num_p <- sum of num`, `den_p <- sum of den` (bincount with weights). Rows outside the list are dropped.
3. Sensitivity: one row adds (num, den) to one partition, at most `U_num + U_den` in L1. Disjoint partitions make all of them one vector (parallel composition), so the cost does not grow with the number of partitions. Replace-one can move the row to another partition: `SENS <- 2·(U_num + U_den)`. With a negative lower bound use the largest absolute value of each range: `2·(max(|L_num|, |U_num|) + max(|L_den|, |U_den|))`.
4. `noisy <- [num_p..., den_p...] + laplace(0, SENS/EPS)` per entry.
5. Post-processing: `ratio_p <- clip(num_p / max(den_p, FLOOR), range)`; relative error by the delta method `√2·scale·sqrt(1 + ratio²) / (den_p · ratio)`; flag partitions above the public maximum. Larger levels (regions) are sums of noisy partitions, also free.

## Privacy accounting
One Laplace release of the stacked vector with L1 sensitivity SENS under replace-one: `epsilon_spent <- SENS / scale`, assert <= EPS. Ratios, flags and aggregates to coarser levels are post-processing.

## Pitfalls
- Spending EPS on each partition and adding them up (wasteful), or charging one partition's EPS when partitions overlap (wrong).
- The noise is the same in every partition, so small partitions get ratios that are mostly noise; flag them with a fixed rule on noisy values, never drop them using exact sizes.
- A partition list from `unique()` of the data: which partitions exist can be private.
- Aggregating to larger areas from exact sums next to the noisy ones.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
