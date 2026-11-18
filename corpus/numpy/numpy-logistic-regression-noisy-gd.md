---
id: numpy-logistic-regression-noisy-gd
title: Logistic regression by clipped noisy gradient descent (DP-GD, zCDP accounting) in NumPy
framework: numpy
framework_version: "any"
mechanism: gaussian
task_type: classification
dp_notion: approx
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.normal
  - numpy.clip
  - numpy.linalg.norm
  - numpy.exp
source_url: https://arxiv.org/abs/1607.00133
source_license: original code (algorithm from cited paper)
---

## Goal
Train a binary logistic-regression classifier with (ε, δ)-DP using full-batch gradient descent with per-record gradient clipping and Gaussian noise (the clip-and-noise core of DP-SGD, Abadi et al., CCS 2016), accounted with zCDP (Bun & Steinke, TCC 2016, Props. 1.3 and 1.6, Lemma 2.3). Neighbouring relation: replace-one (n public).

## Steps
1. Public feature bounds; centre each feature on its range midpoint and scale to [−1, 1]; clamp; add an intercept column. Label y ∈ {0, 1} from a public positive value. N public (replace-one).
2. ρ from (ε, δ): √ρ = √(ln(1/δ) + ε) − √(ln(1/δ)). Public T, clip norm C, learning rate.
3. σ = Δ·√(T/(2ρ)) with Δ = 2C (replace-one; C under add/remove, but then N is private and must not be used as a divisor without its own noise).
4. Each step: p = 1/(1 + exp(−Xw)); per-record gradient g_i = (p_i − y_i)·x_i; clip each to norm C; `w <- w − lr·(Σ g_i + normal(0, σ, d)) / N`.
5. Release w. Accuracy on a held-out set is a separate question: computing it on private data and publishing it costs extra budget.

## Privacy accounting
ρ_total = T·Δ²/(2σ²) = ρ; ε_spent = ρ + 2√(ρ ln(1/δ)); assert ≤ ε.

## Pitfalls
- Seeds: a fixed generator seed makes the noise predictable; use `numpy.random.default_rng()` with no argument.
- Clipping after summing.
- Standardising features with the data's mean and standard deviation (not public); use public ranges.
- Too many steps at small N: noise variance grows with T.
- Reporting training accuracy computed exactly on the private data as part of the DP release.
- Objective or output perturbation (`numpy-logistic-regression-objective-perturbation`, `numpy-logistic-regression-output-perturbation`) give pure ε-DP, but require the exact minimiser.
