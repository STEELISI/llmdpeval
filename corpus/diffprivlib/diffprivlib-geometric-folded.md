---
id: diffprivlib-geometric-folded
title: Integer statistic kept in range by folding with GeometricFolded
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.GeometricFolded
  - diffprivlib.mechanisms.GeometricTruncated
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/geometric.py
source_license: MIT
---

## Goal
Release an integer statistic (a count or integer total) whose valid range [L, U] is public, so that the output always lies in that range, using discrete noise and pure epsilon-DP. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

Signature: `GeometricFolded(*, epsilon, sensitivity=1, lower, upper, random_state=None)`

## Steps
1. Compute the exact integer statistic and state its sensitivity (1 for a count; U_v - L_v for a sum of integers clipped to [L_v, U_v] under replace-one).
2. Public output range `[L, U]`; bounds must be integers or half-integers (e.g. `lower=0.5` excludes 0).
3. `mech <- mechanisms.GeometricFolded(epsilon=?, sensitivity=SENS, lower=L, upper=U)` (keyword-only).
4. `dp_value <- mech.randomise(true_value)` (an int). Noise that lands outside [L, U] is reflected back about the nearest bound.
5. `acc.spend(epsilon, 0)` on a `BudgetAccountant` (mechanisms do not charge it).
6. Output the true value, DP value, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced value moves the total by at most U - L, the sensitivity passed. Add/remove alternative (n private): use max(|L|, |U|) as the sensitivity. Folding is post-processing of the geometric output, so the cost is the epsilon passed, valid only if `sensitivity` is right. Record it with `acc.spend(epsilon, 0)`.

## Pitfalls
- Folding vs truncation: `GeometricTruncated` piles all outside mass onto the bound (many outputs exactly equal to L or U); `GeometricFolded` reflects it (smoother near the bound). Both are biased near the bounds; `mech.bias(value)` reports it.
- `diffprivlib.models.KMeans` uses `GeometricFolded(lower=0.5, upper=inf)` internally for cluster counts, so counts are at least 1.
- The input `value` must be an integer; pass `int(...)`.
- Range bounds from the data (true min/max) are a leak.
