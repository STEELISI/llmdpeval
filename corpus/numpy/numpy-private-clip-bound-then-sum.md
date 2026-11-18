---
id: numpy-private-clip-bound-then-sum
title: Choosing a clipping bound privately, then a bounded sum, in NumPy
framework: numpy
framework_version: "any"
mechanism: exponential
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.choice
  - numpy.random.Generator.laplace
  - numpy.searchsorted
  - numpy.clip
  - numpy.exp
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release the sum of a non-negative column whose upper bound is not known in advance, with ε-DP. Spend part of ε to pick a clipping bound C from a public log-spaced grid with the exponential mechanism (a high quantile, e.g. q = 0.95), then clamp to [0, C] and release a Laplace sum with the rest. Exponential mechanism (Dwork & Roth 2014, Thm. 3.10), Laplace (Thm. 3.6), sequential composition (Cor. 3.15). Neighbouring relation: replace-one (n public).

## Steps
1. Public candidate grid for C, e.g. powers of 2 up to a public ceiling; a public target quantile q.
2. Split ε = ε_bound + ε_sum (e.g. 0.2 / 0.8).
3. Pick C: utility u(c) = −|#{x ≤ c} − q·N| (sensitivity 1); sample with logits ε_bound·u/2.
4. Clamp: `x_c <- numpy.clip(x, 0, C)`. C is now a DP output, so the second step may depend on it. The sum's sensitivity is C under both add/remove and replace-one, because values lie in [0, C].
5. `noisy_sum <- sum(x_c) + rng.laplace(loc=0, scale=C/ε_sum)`.

## Privacy accounting
Two mechanisms on the same data: ε_bound + ε_sum (adaptive sequential composition; the second mechanism's parameter comes from the first one's output, which is allowed). Compute both parts in code and assert the total.

## Pitfalls
- C from the true maximum or a true percentile: data-dependent sensitivity, no DP guarantee.
- Reusing C as "public" in later analyses without counting ε_bound.
- Choosing q = 1 (the max): the exponential mechanism then targets the maximum, which is unstable; a high quantile under-clips a little but is far more stable.
- Signed data: use a symmetric grid on |x| and sensitivity C (replace-one: 2C).
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
