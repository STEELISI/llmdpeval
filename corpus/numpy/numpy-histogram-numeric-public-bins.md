---
id: numpy-histogram-numeric-public-bins
title: Histogram of a numeric column over public bin edges with the Laplace mechanism in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.histogram
  - numpy.clip
  - numpy.rint
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release counts of a numeric column in fixed bins with ε-DP. The bin edges are public; every record falls in exactly one bin, so the count vector is a histogram query (Dwork & Roth 2014, Example 3.2) released with the Laplace mechanism (Def. 3.3, Thm. 3.6). Default neighbours: replace-one (bounded DP, n public), Δ₁ = 2.

## Steps
1. Public edges E = e_0 < e_1 < ... < e_B, chosen from domain knowledge.
2. Decide what happens to out-of-range values, per record: clamp them into the edge bins, or drop them. Both are per-record operations, so sensitivity is unchanged.
3. `counts <- numpy.histogram(x, bins=E)[0]` (always pass explicit edges).
4. L1 sensitivity Δ₁ of the count vector: 1 under add/remove; 2 under replace-one (one bin −1, another +1). Same with a public, fixed N ("sized" data): sizing does not lower a histogram's replace-one sensitivity below 2.
5. `noisy <- counts + rng.laplace(loc=0, scale=Δ₁/ε, size=B)`.
6. Post-process if wanted: `numpy.rint`, clip at 0. Both are free.

## Privacy accounting
Replace-one (bounded DP, n public) by default, Δ₁ = 2 (1 only under add/remove). The whole vector costs ε once (one vector query with L1 sensitivity Δ₁). Compute epsilon_spent = Δ₁/scale and assert it ≤ ε.

## Pitfalls
- `numpy.histogram(x, bins=10)`: the edges come from the data's min and max, which leaks them and changes the bins between neighbours.
- Δ₁ = 1 under replace-one.
- Charging ε per bin or splitting ε across bins: bins are disjoint.
- Releasing only the non-empty bins: which bins are empty is itself information.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
