---
id: numpy-logistic-regression-objective-perturbation
title: Logistic regression by objective perturbation (Chaudhuri-Monteleoni-Sarwate Alg. 2) in NumPy
framework: numpy
framework_version: "any"
mechanism: objective-perturbation
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.gamma
  - numpy.random.Generator.normal
  - numpy.linalg.solve
  - numpy.linalg.norm
  - numpy.log
  - numpy.exp
source_url: https://jmlr.org/papers/v12/chaudhuri11a.html
source_license: original code (algorithm from cited paper)
---

## Goal
Train L2-regularised logistic regression with pure ε-DP by adding a random linear term to the objective and releasing its exact minimiser. Chaudhuri, Monteleoni & Sarwate, JMLR 12 (2011), Algorithm 2 and Theorem 9. Relation: replace one record, N public. Requires ‖x‖ ≤ 1 and a loss with |ℓ''| ≤ c (c = 1/4 for logistic loss).

## Steps
1. Scale rows so ‖x‖ ≤ 1 exactly (public bounds → [−1, 1] per feature, intercept column, divide by √(D+1)). Labels in {−1, +1}. Public λ > 0.
2. Slack: ε' = ε − log(1 + 2c/(Nλ) + c²/(N²λ²)).
3. If ε' > 0: extra ridge Δ = 0. Otherwise Δ = c / (N·(e^{ε/4} − 1)) − λ and ε' = ε/2.
4. Noise b with density ∝ exp(−(ε'/2)·‖b‖): norm `rng.gamma(shape=d, scale=2/ε')`, uniform direction.
5. Minimise J(w) = (1/N)Σ log(1 + exp(−y_i wᵀx_i)) + (λ/2)‖w‖² + (1/N)·bᵀw + (Δ/2)‖w‖² exactly (Newton's method).
6. Release the minimiser.

## Privacy accounting
ε_spent = ε' + 2·log(1 + c/(N(λ + Δ))). In both branches of step 3 this equals ε; compute it in code from the actual ε', λ, Δ and N and assert it. The slack term uses ‖x‖ ≤ 1. With rows of norm q > 1 the curvature term grows with q², so keep q = 1 instead of patching the formula.

## Pitfalls
- Rows with norm above 1 (feature scaling to [−1, 1] alone is not enough).
- Approximate minimisation: the theorem assumes the exact minimiser; iterate Newton to convergence.
- Forgetting the 1/N on the noise term when the loss is averaged.
- λ chosen on the private data.
- Fitting an unregularised intercept separately: it must be part of w and of ‖x‖.
