---
id: numpy-discrete-gaussian
title: Discrete Gaussian noise for integer counts (Canonne-Kamath-Steinke sampler) in NumPy
framework: numpy
framework_version: "any"
mechanism: gaussian
task_type: histogram
dp_notion: zcdp
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.geometric
  - numpy.random.Generator.random
  - numpy.exp
  - numpy.floor
source_url: https://arxiv.org/abs/2004.00010
source_license: original code (algorithm from cited paper)
---

## Goal
Release integer counts (a histogram over public categories) with integer-valued Gaussian noise, giving ρ-zCDP and hence (ε, δ)-DP. Discrete Gaussian: Canonne, Kamath & Steinke, NeurIPS 2020 (arXiv 2004.00010): ρ = Δ₂²/(2σ²) (their Thm. 4), sampled by rejection from a discrete Laplace (their Alg. 3). Default neighbours: replace-one (bounded DP, n public), Δ₂ = √2.

## Steps
1. Public, fixed category list; integer counts, zeros included. L2 sensitivity Δ₂ = 1 (add/remove) or √2 (replace-one).
2. ρ from (ε, δ): √ρ = √(ln(1/δ) + ε) − √(ln(1/δ)); σ = Δ₂/√(2ρ).
3. Discrete Laplace with parameter t (P(y) ∝ exp(−|y|/t)): difference of two i.i.d. `rng.geometric(p)` draws with p = 1 − exp(−1/t).
4. Discrete Gaussian sampler: t = floor(σ) + 1; repeat: draw Y from discrete Laplace(t); accept with probability exp(−(|Y| − σ²/t)² / (2σ²)).
5. `noisy <- counts + one sample per category`. The output stays an integer.

## Privacy accounting
Replace-one (bounded DP, n public) by default: Δ₂ = √2 (1 only under add/remove). ρ_spent = Δ₂²/(2σ²); ε_spent = ρ + 2√(ρ·ln(1/δ)), asserted ≤ ε. Integer outputs avoid the floating-point problems of continuous noise on its output (Mironov 2012).

## Pitfalls
- This NumPy sampler uses floating-point coin flips, so it is not the exact sampler from the paper (which uses exact rational arithmetic); it is fine for evaluation, not for a hardened deployment.
- Rounding continuous Gaussian noise to integers and calling it discrete Gaussian: a different distribution, with no exact guarantee.
- σ² vs σ: the acceptance test uses σ².
- Using L1 sensitivity.
