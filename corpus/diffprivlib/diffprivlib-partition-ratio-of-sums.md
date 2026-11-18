---
id: diffprivlib-partition-ratio-of-sums
title: Sums and a ratio of two sums for many disjoint partitions (small areas) with diffprivlib's Laplace mechanism
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.mechanisms.Laplace.randomise
  - diffprivlib.mechanisms.Laplace.variance
  - diffprivlib.accountant.BudgetAccountant
  - numpy.bincount
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/laplace.py
source_license: MIT
---

## Goal
Release, for every partition in a public list (areas, sites, units; possibly thousands), the sum of two bounded non-negative columns and their ratio, and flag partitions where the noise dominates. Replace-one neighbours (bounded DP, n public).

## Steps
1. Public inputs: the partition list, bounds `(0, U_num)` and `(0, U_den)`, the valid ratio range, a denominator floor and a maximum relative error (fixed rules).
2. Clamp both columns; per partition sums with `bincount(codes, weights, minlength=P)`. Rows outside the list are dropped.
3. Sensitivity: one row adds (num, den) to one partition, at most `U_num + U_den` in L1; disjoint partitions form one vector release. Replace-one can move the row to another partition: `SENS <- 2·(U_num + U_den)`.
4. `mech <- Laplace(epsilon=EPS, sensitivity=SENS)`; `noisy <- [mech.randomise(v) for v in stacked sums]` (an L1 vector release is this noise on every entry); `acc.spend(EPS, 0)` once. Noise scale: `sqrt(mech.variance(0) / 2)`.
5. Post-processing: ratio with the public floor, clipped to its range; relative error by the delta method; flag partitions above the maximum.

## Privacy accounting
One release of the stacked vector with L1 sensitivity SENS under replace-one, recorded once: `epsilon_spent <- acc.total()[0]`, assert <= EPS.

## Pitfalls
- A loop of `tools.sum` per partition on each partition's own rows: not DP at any share, because diffprivlib 0.6.6 truncates the output to a range set by the private partition size (an empty partition always returns 0). Even on full-length masked arrays, under replace-one a row can touch two partitions, so each call at EPS costs 2·EPS. One vector release avoids both.
- `tools.mean` per partition treats the partition size as public.
- Dropping small partitions with exact sizes; use the noisy flag.
- A partition list from the data.
