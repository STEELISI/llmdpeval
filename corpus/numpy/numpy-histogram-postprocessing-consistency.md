---
id: numpy-histogram-postprocessing-consistency
title: Making a noisy histogram consistent (non-negative, integer, fixed total) by post-processing in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.sort
  - numpy.cumsum
  - numpy.maximum
source_url: https://arxiv.org/abs/0904.0942
source_license: original code (algorithm from cited paper)
---

## Goal
Release a histogram over public categories with ε-DP whose counts are non-negative and add up to a total, without spending extra budget. Post-processing (Dwork & Roth 2014, Prop. 2.1) of a Laplace histogram; consistency by projection follows Hay, Rastogi, Miklau & Suciu, VLDB 2010 (arXiv 0904.0942). Default neighbours: replace-one (bounded DP, n public), L1 sensitivity 2.

## Steps
1. Laplace histogram over a public list (L1 sensitivity 1 add/remove, 2 replace-one); see `numpy-laplace-histogram-public-categories`.
2. Choose the target total T: the public N if N is public by design (replace-one), otherwise the sum of the noisy counts (itself post-processing). Never the exact private N under add/remove.
3. Project the noisy vector onto {c ≥ 0, Σc = T} in least squares (Euclidean projection onto a scaled simplex): sort descending, find the largest j with u_j − (Σ_{i≤j} u_i − T)/j > 0, set θ = (Σ_{i≤j} u_i − T)/j, and return max(noisy − θ, 0). If T ≤ 0 (a noisy total on small data), return all zeros instead: no j qualifies.
4. Optional integer rounding that keeps the total: floor everything, then give the remaining units to the entries with the largest fractional parts.

## Privacy accounting
Replace-one (bounded DP, n public) by default: Δ₁ = 2 (1 only under add/remove, n private). Only step 1 touches the data: ε_spent = Δ₁/scale. Steps 2-4 use only the noisy vector and public constants, so they cost nothing. Compute and assert ε_spent in code.

## Pitfalls
- Using the exact count of records as T when N is not public.
- Clipping negatives to 0 without re-normalising: the total drifts upward (positive bias for small bins).
- Re-noising after projection "to be safe": wasted budget.
- Rounding first and projecting later: the total breaks again.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
