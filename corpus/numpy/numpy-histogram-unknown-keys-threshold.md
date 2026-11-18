---
id: numpy-histogram-unknown-keys-threshold
title: Counts for categories not known in advance - Laplace plus a release threshold (epsilon, delta) in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: approx
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.log
  - numpy.unique
source_url: https://arxiv.org/abs/1909.01917
source_license: original code (algorithm from cited paper)
---

## Goal
Release counts per category when the category list cannot be fixed in advance (free text, IDs), with (ε, δ)-DP. Noise every observed category's count and release only those above a threshold τ. Each record must contribute to exactly one category. Default neighbours: replace-one (bounded DP, n public), so two keys can change. Korolova et al., WWW 2009; Wilson et al., "Differentially Private SQL with Bounded User Contribution", PoPETs 2020 (arXiv 1909.01917).

## Steps
1. Observed keys and counts: `keys, counts <- numpy.unique(x, return_counts=True)`.
2. Laplace noise per observed key, replace-one: a changed record lowers one key and raises another, so Δ₁ = 2: `noisy <- counts + rng.laplace(loc=0, scale=2/ε, size=len(keys))`.
3. Threshold τ = 1 + (2/ε)·ln(1/δ) (each of the two keys that can appear or vanish gets δ/2). Release (key, noisy count) only where noisy > τ.
4. Choose δ much smaller than 1/N (e.g. 1e-6).

## Privacy accounting
Replace-one (bounded DP, n public): Δ₁ = 2, scale 2/ε, τ = 1 + (2/ε)·ln(1/δ). For keys present in both neighbours the Laplace part gives ε. The only extra risk is a key held by one record (count 1) that exists in one neighbour only: it passes the threshold with probability ½·exp(−(τ − 1)·ε/2) = δ/2, and at most two such keys change, so the release is (ε, δ)-DP. Compute τ in code from ε and δ, and return both. Add/remove alternative (one contribution per record): scale 1/ε and τ = 1 + ln(1/(2δ))/ε.

## Pitfalls
- Releasing every observed key with noise but no threshold: a key seen once is revealed with certainty.
- Threshold picked by eye ("keep counts above 5") with no δ calculation.
- Records that contribute to several keys: cap contributions per record first and multiply the sensitivity by the cap.
- If the category list IS public, use `numpy-laplace-histogram-public-categories` instead: pure ε, no threshold, zeros reported.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
