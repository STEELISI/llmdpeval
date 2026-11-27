---
id: numpy-fraction-of-rows-public-n
title: Fraction of rows matching a predicate (support, classifier accuracy) as a Laplace count over a public n in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: count
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: "Dwork & Roth 2014 textbook, author-hosted PDF; definitions paraphrased, no text copied"
---

## Goal
Release the share of rows satisfying a row-level predicate (support of a rule, accuracy of a fixed classifier on private test rows, a hit rate) with pure ε-DP, when the number of rows N is public (bounded DP, replace-one).

## Steps
1. Indicator per row from that row only (`pred == label`, rule matches, ...). The classifier must not be fit on these rows, or must itself be DP.
2. `count <- sum(INDICATORS)`. Under replace-one with public N the count's sensitivity is 1 (one replaced row flips at most one indicator).
3. `scale <- 1 / EPS`; `noisy <- count + rng.laplace(0, scale)` with `rng <- numpy.random.default_rng()` (no seed).
4. `fraction <- clip(noisy / N, 0, 1)`: post-processing.

## Privacy accounting
NumPy has no accountant: epsilon_spent = sensitivity / scale = 1 / scale. State the relation (replace-one, public N) next to it and assert epsilon_spent <= EPS.

## Pitfalls
- Sensitivity 1/N applied to the fraction is the same release; do not add noise to both the count and the fraction.
- If N is not public, the fraction is a ratio of two private quantities: release a noisy count of all rows too (split ε) or use a public N.
- Printing the exact fraction next to the private one.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
