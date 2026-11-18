---
id: numpy-topk-from-noisy-histogram
title: Top-k categories by count from one noisy histogram (post-processing) in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.argsort
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release the k most frequent categories of a public category list with ε-DP by releasing a full Laplace histogram once and sorting it. Laplace mechanism on a vector query (Dwork & Roth 2014, Def. 3.3, Thm. 3.6) plus post-processing (Prop. 2.1). Valid for COUNT scores, where one record touches one bin. Default neighbours: replace-one (bounded DP, n public), Δ₁ = 2.

## Steps
1. Public, fixed category list (K entries); exact counts for all of them, zeros included.
2. L1 sensitivity of the whole count vector: Δ₁ = 1 under add/remove, 2 under replace-one (one bin −1, another +1).
3. `noisy <- counts + rng.laplace(loc=0, scale=Δ₁/ε, size=K)`: one release, cost ε.
4. `top <- first k of numpy.argsort(-noisy)`; this and any other use of `noisy` is free.

## Privacy accounting
Replace-one (bounded DP, n public) by default: Δ₁ = 2 (1 only under add/remove). One Laplace release of a vector with L1 sensitivity Δ₁ at scale Δ₁/ε is ε-DP; sorting is post-processing. The scale does not grow with k, unlike peeling (scale 2k/ε for non-monotone scores). Peeling and one-shot Gumbel release less (only names), so prefer them when K is large and k small; this route tends to win when K is small. Compute epsilon_spent = Δ₁/scale and assert it.

## Pitfalls
- Using this for scores where one record touches many entries (sums over many groups, rates): then Δ₁ is the total change over all entries; for rates both numerator and denominator move.
- Δ₁ = 1 under replace-one.
- Building the category list from the data's distinct values.
- Calling sorted Laplace noise "one-shot top-k" and then re-noising: noise once, then only post-process.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
