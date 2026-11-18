---
id: numpy-linear-regression-sufficient-stats
title: DP linear regression by noisy sufficient statistics (Laplace) in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: regression
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.clip
  - numpy.column_stack
  - numpy.ones
  - numpy.triu_indices
  - numpy.eye
  - numpy.linalg.solve
source_url: https://arxiv.org/abs/1803.02596
source_license: "Wang 2018 (UAI) arXiv preprint; method described, sensitivity derived from Dwork & Roth Def. 3.1"
---

## Goal
Fit ordinary least squares with ε-DP by releasing noisy XᵀX and Xᵀy and solving from them: sufficient statistics perturbation (SSP; Vu & Slavkovic 2009, Foulds et al. 2016, analysed in Wang 2018). Laplace version: noise calibrated to L1 sensitivity (Dwork & Roth Def. 3.1, Thm. 3.6). Neighbouring relation: replace-one (n public).

## Steps
1. Public bounds per feature and for the label, fixed before seeing the data: |x_j| ≤ B_j, |y| ≤ B_y. If a public range [L, U] is far from 0, centre on its midpoint first (B = half-width). Clamp: `X <- numpy.clip(X, -B, B)`, `y <- numpy.clip(y, -B_y, B_y)`.
2. Intercept column: `Xa <- numpy.column_stack([numpy.ones(N), X])`, bound 1 for that column; Ba = [1, B_1, ..., B_D].
3. Exact statistics: A = Xaᵀ·Xa (symmetric), v = Xaᵀ·y. Release only A's upper triangle: `a <- A[numpy.triu_indices(D + 1)]`.
4. L1 sensitivity under **replace-one** (one row removed, another added; each row contributes x·xᵀ to A and x·y to v):
   - upper triangle of A: Δ_A = 2·Σ_{j ≤ k} Ba_j·Ba_k;
   - v: Δ_v = 2·B_y · Σ_j Ba_j.
   (Add/remove alternative: half of each.)
5. Split ε_A + ε_v = ε; `a_n <- a + rng.laplace(loc=0, scale=?, size=len(a))` with scale = Δ_A/ε_A; `v_n <- v + rng.laplace(loc=0, scale=?, size=D + 1)` with scale = Δ_v/ε_v.
6. Rebuild symmetric A_n from a_n (fill upper triangle, mirror). Solve `beta <- numpy.linalg.solve(A_n + LAM * numpy.eye(D + 1), v_n)` with a public ridge LAM > 0; the noisy matrix may be singular or indefinite. All of step 6 is post-processing.
7. If features were centred in step 1, apply the same shift to new inputs before predicting.

## Privacy accounting
Total = ε_A + ε_v (sequential composition, Dwork & Roth Cor. 3.15). Both sensitivities are the replace-one values. Assert the sum; print the bounds, both sensitivities and the relation. Predictions on new public inputs are free (Prop. 2.1). Wang 2018 gives a Gaussian (ε, δ) variant with an adaptive ridge (AdaSSP) if δ > 0 is allowed.

## Pitfalls
- Bounds from the training data's min/max: leaks and voids the sensitivity bound.
- Noising the full matrix but computing Δ_A for one triangle (each off-diagonal entry appears twice in the full matrix).
- Forgetting the intercept column in both sensitivities.
- No ridge: solving a noisy near-singular matrix gives wild coefficients.
- Fitting with a non-private solver and adding noise to the coefficients without a coefficient-sensitivity analysis.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
