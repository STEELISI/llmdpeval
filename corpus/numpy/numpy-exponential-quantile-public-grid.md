---
id: numpy-exponential-quantile-public-grid
title: Quantile or median over a public grid of candidate values with the exponential mechanism in NumPy
framework: numpy
framework_version: "any"
mechanism: exponential
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.choice
  - numpy.searchsorted
  - numpy.sort
  - numpy.exp
source_url: https://doi.org/10.1109/FOCS.2007.66
source_license: original code (algorithm from cited paper)
---

## Goal
Release the q-th quantile (median: q = 0.5) of a numeric column with ε-DP by choosing one value from a PUBLIC grid of candidates (e.g. every integer in a public range, or a log-spaced grid). Exponential mechanism: McSherry & Talwar 2007; Dwork & Roth 2014, Thm. 3.10. Neighbouring relation: replace-one (n public).

## Steps
1. Public sorted grid G = g_1 < ... < g_m, fixed from domain knowledge (never from the data).
2. Rank of each candidate: `below[i] <- numpy.searchsorted(numpy.sort(x), G[i], side="right")` = number of records ≤ g_i.
3. Utility u_i = −|below[i] − q·N|. Its sensitivity is 1: one record moves `below` by at most 1 (replace-one), and under add/remove `below` moves by 0 or 1 while q·N moves by q, so the difference moves by at most max(q, 1−q) ≤ 1.
4. Logits ε·u_i / (2·1); subtract the max before `numpy.exp`; normalise; `idx <- rng.choice(m, p=probs)`.
5. Release G[idx]. Several quantiles: split ε across them.

## Privacy accounting
One exponential-mechanism draw with Δu = 1 costs ε; k quantiles at ε/k each sum to ε (sequential composition, Dwork & Roth Cor. 3.15). Compute and assert the total in code.

## Pitfalls
- A grid built from the observed values: leaks them. The interval version (`numpy-exponential-quantile`) is the right tool when you want outputs between data points.
- Too coarse a grid caps accuracy; a very wide grid spends probability mass on empty regions.
- Under add/remove, N inside the utility is fine (sensitivity stays ≤ 1); do not divide the utility by N.
- Exponentiating without subtracting the max: overflow for large ε·N.
