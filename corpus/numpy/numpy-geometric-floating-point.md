---
id: numpy-geometric-floating-point
title: Floating-point caveat (Mironov) and the geometric mechanism for integer queries in NumPy
framework: numpy
framework_version: "any"
mechanism: geometric
task_type: count
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.geometric
  - numpy.expm1
source_url: https://arxiv.org/abs/0811.2841
source_license: "Ghosh, Roughgarden & Sundararajan (STOC 2009) arXiv preprint; mechanism paraphrased, no code copied"
---

## Goal
Release an integer-valued query (a count, a histogram bin) with ε-DP using integer noise, which avoids the floating-point attack on textbook Laplace noise. Geometric mechanism: Ghosh, Roughgarden & Sundararajan 2009. Caveat: Mironov, "On significance of the least significant bits for differential privacy", CCS 2012. Laplace noise drawn as doubles (e.g. via inverse CDF on a uniform double) cannot produce every double, and which low-order bits occur depends on the true value. Mironov 2012 showed that an attacker who sees the exact noisy double can often tell neighbouring inputs apart, so the real guarantee can be far weaker than ε. His fix is the snapping mechanism (clamp and round the output to a grid). Rounding the output of a Laplace draw by itself is not his fix. For integer queries, integer noise sidesteps the issue. Neighbours: replace-one (bounded DP, n public); a single count has Δ = 1 there, a histogram vector Δ = 2.

## Steps
1. Integer query value `t` and integer sensitivity Δ under the stated relation (count: Δ = 1 under add/remove and replace-one; histogram vector: Δ = 1 add/remove, 2 replace-one).
2. Two-sided geometric noise as the difference of two geometric draws:
   p = 1 − e^(−ε/Δ), computed as `p <- -numpy.expm1(-ε/Δ)` (accurate for small ε/Δ);
   `noise <- rng.geometric(p, size=?) - rng.geometric(p, size=?)` (two independent draws, same shape).
   The difference has P(noise = k) ∝ e^(−ε·|k|/Δ) for integer k, the two-sided geometric distribution of Ghosh et al.
3. `dp_value <- t + noise` (still an integer). Post-process as usual: clip at 0, etc. (free).

## Privacy accounting
Replace-one (bounded DP, n public): the single count in the skeleton has Δ = 1 (one changed row flips at most one indicator); a full histogram released this way needs Δ = 2. Two-sided geometric noise with parameter e^(−ε/Δ) on an integer query of L1 sensitivity Δ is (ε, 0)-DP, the discrete analogue of Laplace with scale Δ/ε (Ghosh et al. 2009). Compose with other releases as usual (sum of epsilons). Print ε, Δ and p.

## Pitfalls
- Using p = ε/Δ or p = e^(−ε/Δ): p is the success probability 1 − e^(−ε/Δ).
- Reusing one geometric draw for both halves: the difference is then 0.
- `numpy.random.Generator.geometric` counts trials, so its support starts at 1; the difference of two draws cancels that shift, a single draw does not.
- NumPy still samples through floating point internally, so this reduces but does not formally eliminate the precision issue; exact samplers exist (Canonne, Kamath & Steinke 2020, "The Discrete Gaussian for Differential Privacy").
- Non-integer queries (means, sums of reals) need snapping or a vetted library, not this outline.
