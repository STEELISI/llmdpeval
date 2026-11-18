---
id: numpy-quantiles-from-noisy-histogram
title: Several quantiles from one noisy histogram over public bins in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.histogram
  - numpy.clip
  - numpy.cumsum
  - numpy.searchsorted
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release many quantiles of a numeric column (quartiles, deciles) for the price of one ε by noising a histogram over PUBLIC bin edges once and reading all quantiles off its cumulative sum. Laplace histogram (Dwork & Roth 2014, Example 3.2, Thm. 3.6) plus post-processing (Prop. 2.1). Default neighbours: replace-one (bounded DP, n public), Δ₁ = 2.

## Steps
1. Public bin edges E (from domain knowledge: a public range and a bin width); clamp values into [E_0, E_last] so every record falls in some bin.
2. `counts <- numpy.histogram(x_clamped, bins=E)[0]`.
3. L1 sensitivity Δ₁: 1 (add/remove), 2 (replace-one).
4. `noisy <- counts + rng.laplace(loc=0, scale=Δ₁/ε, size=len(counts))`.
5. Post-process: `noisy <- numpy.clip(noisy, 0, None)`; `cdf <- numpy.cumsum(noisy) / sum(noisy)`.
6. For each q: `b <- numpy.searchsorted(cdf, q)`; return the upper edge of bin b (or interpolate linearly inside the bin).

## Privacy accounting
Replace-one (bounded DP, n public) by default: Δ₁ = 2 (1 only under add/remove). One histogram release costs ε; every quantile read from it is free. Compare with the exponential mechanism, which pays ε/k per quantile. Compute epsilon_spent = Δ₁/scale in code and assert it.

## Pitfalls
- Bin edges from the data's min/max or from np.histogram's automatic bins (`bins=10` uses the data range): leaks the extremes.
- Dropping out-of-range values: fine for privacy, but quantiles then refer to the clipped population; clamping keeps N.
- Many narrow bins: noise accumulates in the cumulative sum (error grows like √bins·scale). Choose widths for the accuracy you need.
- Releasing `counts` instead of `noisy` anywhere.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
