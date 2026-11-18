---
id: numpy-laplace-error-bound
title: Reporting a DP answer with its noise error bound (Laplace tail) in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.clip
  - numpy.log
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release a bounded sum with ε-DP together with an interval that contains the noise-free (clamped) answer with probability 1 − β. The interval comes from the Laplace tail bound P(|Lap(b)| ≥ t·b) = e^{−t} (Dwork & Roth 2014, Fact 3.7), so it depends only on public quantities and costs nothing. Neighbouring relation: replace-one (n public).

## Steps
1. Public bounds [L, U]; clamp; Laplace noise with scale b = Δ/ε (replace-one: Δ = U − L; alternative add/remove: Δ = max(|L|, |U|)).
2. Half-width of the (1 − β) interval: t = b·ln(1/β). For β = 0.05: t ≈ 3·b.
3. Release `(noisy, noisy − t, noisy + t)`, and state β.
4. Use the same bound to sanity-check code: over many runs, about β of them should miss the true clamped answer; if almost none miss, the noise is smaller than claimed.

## Privacy accounting
Replace-one, Δ = U − L; ε_spent = Δ/b, computed in code. The interval is a function of the noisy answer and public constants (b, β): post-processing, no extra cost.

## Pitfalls
- Computing the interval from the data (e.g. adding the sample standard deviation): leaks and is not a DP release.
- The interval covers the noise only. Clamping bias (values outside [L, U]) is a separate, one-sided error the interval does not include.
- For k independent answers, a joint interval needs β/k per answer (union bound).
- Gaussian noise has a different tail: t = σ·√(2·ln(2/β)) is a valid two-sided bound.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
