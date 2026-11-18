---
id: numpy-logistic-regression-output-perturbation
title: Logistic regression by output perturbation (Chaudhuri-Monteleoni-Sarwate) in NumPy
framework: numpy
framework_version: "any"
mechanism: output-perturbation
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.gamma
  - numpy.random.Generator.normal
  - numpy.linalg.solve
  - numpy.linalg.norm
  - numpy.clip
source_url: https://jmlr.org/papers/v12/chaudhuri11a.html
source_license: original code (algorithm from cited paper)
---

## Goal
Train L2-regularised logistic regression with pure ε-DP by computing the exact minimiser and adding a noise vector whose norm follows a Gamma distribution. Chaudhuri, Monteleoni & Sarwate, JMLR 12 (2011), Algorithm 1 (output perturbation), sensitivity 2/(Nλ) for 1-Lipschitz losses with ‖x‖ ≤ 1 (Corollary 8). Neighbouring relation: replace one record, N public.

## Steps
1. Public feature bounds → centre and scale to [−1, 1], add an intercept column, then divide every row by √(D+1) so that ‖x‖ ≤ 1 exactly. Labels y ∈ {−1, +1}.
2. Objective J(w) = (1/N)·Σ log(1 + exp(−y_i·wᵀx_i)) + (λ/2)·‖w‖² with a public λ > 0.
3. Minimise to convergence (Newton's method: gradient and Hessian in closed form, `numpy.linalg.solve`). The privacy proof assumes the exact minimiser.
4. L2 sensitivity of the minimiser: Δ = 2/(N·λ).
5. Noise b with density ∝ exp(−ε‖b‖/Δ): norm `rng.gamma(shape=d, scale=Δ/ε)`, direction `z/‖z‖` with `z <- rng.normal(size=d)`.
6. Release w* + b.

## Privacy accounting
ε-DP for the stated sensitivity; ε_spent = Δ / gamma_scale, computed in code. Δ depends on N, λ and on ‖x‖ ≤ 1: if rows can exceed norm 1 the bound breaks.

## Pitfalls
- Gaussian or per-coordinate Laplace noise with this Δ: the proof is for the Gamma-norm, uniform-direction noise (L2 norm ball).
- Rows not scaled to norm ≤ 1 (scaling each feature to [−1, 1] alone gives norm up to √(D+1)).
- Stopping the optimiser early: the sensitivity argument is about the exact minimiser.
- λ too small: Δ = 2/(Nλ) explodes. λ is a public hyper-parameter; tuning it on the private data costs budget.
- Objective perturbation is usually more accurate at the same ε (`numpy-logistic-regression-objective-perturbation`).
