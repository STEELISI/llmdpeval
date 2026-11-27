---
id: numpy-advanced-composition
title: Many small queries under advanced composition (Dwork-Rothblum-Vadhan) vs basic composition in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: composition
dp_notion: approx
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.sqrt
  - numpy.log
  - numpy.exp
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Answer k counting queries (each sensitivity 1) with Laplace noise under a total (ε, δ) by choosing the per-query ε₀ from the advanced composition theorem when that allows more budget per query than ε/k. Dwork, Rothblum & Vadhan, FOCS 2010; Dwork & Roth 2014, Thm. 3.20. Neighbouring relation: replace-one (n public).

## Steps
1. Public k, ε, δ (δ > 0 is required; with δ = 0 basic composition is optimal, Kairouz, Oh & Viswanath 2015).
2. Advanced bound for k mechanisms that are each ε₀-DP: ε_adv(ε₀) = √(2k·ln(1/δ))·ε₀ + k·ε₀·(e^{ε₀} − 1), with total δ.
3. Find the largest ε₀ with ε_adv(ε₀) ≤ ε by bisection. Compare with ε/k and use the larger.
4. Each query: `answer_i <- exact_i + rng.laplace(loc=0, scale=1/ε₀)`.
5. Report the total as min(k·ε₀, ε_adv(ε₀)) at δ (the basic bound has δ = 0).

## Privacy accounting
Both bounds are valid; report whichever is smaller. Advanced composition only helps when k is large (roughly k > 2·ln(1/δ) for small ε₀): for a handful of queries ε/k wins. Assert the reported total ≤ ε in code.

## Pitfalls
- Using advanced composition with δ = 0 or for a few queries, where it is worse.
- Forgetting that the result is (ε, δ), not pure ε.
- Plugging the total ε into the formula as ε₀.
- For Gaussian noise use zCDP (`numpy-zcdp-gaussian-composition`); it is tighter still.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
