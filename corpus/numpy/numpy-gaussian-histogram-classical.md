---
id: numpy-gaussian-histogram-classical
title: Gaussian histogram with the classical calibration (and when it is invalid) in NumPy
framework: numpy
framework_version: "any"
mechanism: gaussian
task_type: histogram
dp_notion: approx
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.normal
  - numpy.sqrt
  - numpy.log
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release counts over a public category list with (ε, δ)-DP using Gaussian noise and the textbook calibration σ = Δ₂·√(2 ln(1.25/δ)) / ε. Dwork & Roth 2014, Thm. 3.22 (proof in Appendix A); valid only for ε < 1. Default neighbours: replace-one (bounded DP, n public), Δ₂ = √2.

## Steps
1. Public, fixed category list; exact counts, zeros included.
2. L2 sensitivity of the count vector: Δ₂ = 1 under add/remove (one bin moves by 1); √2 under replace-one (one bin −1, another +1).
3. Check 0 < ε < 1. For ε ≥ 1 use the analytic calibration (`numpy-gaussian-mechanism`) or zCDP (`numpy-zcdp-gaussian-composition`).
4. σ = Δ₂·√(2·ln(1.25/δ)) / ε; `noisy <- counts + rng.normal(loc=0, scale=σ, size=K)`.
5. Post-process if wanted (round, clip at 0).

## Privacy accounting
Replace-one (bounded DP, n public) by default: Δ₂ = √2 (1 only under add/remove). Thm. 3.22 gives (ε, δ) for this σ when ε < 1. Recover ε from σ in code (ε = Δ₂·√(2 ln(1.25/δ)) / σ) and assert it. Why Gaussian for histograms: its L2 sensitivity √2 (replace-one) is smaller than the L1 sensitivity 2, and it composes well under zCDP when many histograms are released.

## Pitfalls
- Using the formula with ε ≥ 1: outside the theorem.
- L1 sensitivity in the Gaussian formula, or L2 sensitivity in the Laplace formula.
- `numpy.random.Generator.normal(loc, scale)`: scale is the standard deviation σ, not the variance.
- δ chosen as 1/N or larger.
