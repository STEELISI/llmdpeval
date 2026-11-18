---
id: numpy-linear-regression-ssp-gaussian
title: Multi-feature linear regression by Gaussian sufficient statistics with an adaptive ridge (AdaSSP style) in NumPy
framework: numpy
framework_version: "any"
mechanism: gaussian
task_type: regression
dp_notion: approx
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.normal
  - numpy.clip
  - numpy.column_stack
  - numpy.triu_indices
  - numpy.linalg.eigvalsh
  - numpy.linalg.solve
source_url: https://arxiv.org/abs/1803.02596
source_license: original code (algorithm from cited paper)
---

## Goal
Fit least squares on several features with (ε, δ)-DP by releasing noisy XᵀX and Xᵀy with Gaussian noise, plus a noisy smallest eigenvalue that sets the ridge. Follows the idea of AdaSSP (Wang, UAI 2018, Alg. 2); accounting via zCDP (Bun & Steinke 2016: Gaussian ρ = Δ₂²/(2σ²), ρ adds up, ρ-zCDP gives (ρ + 2√(ρ ln(1/δ)), δ)-DP). Neighbouring relation: replace-one (n public).

## Steps
1. Public bounds per feature and for the label. Centre on each range's midpoint and divide by its half-width, so every feature and the label lie in [−1, 1]; clamp. Add an intercept column of ones. Each augmented row then has ‖x‖² ≤ 1 + D =: B².
2. Budget: convert (ε, δ) to ρ: √ρ = √(ln(1/δ) + ε) − √(ln(1/δ)). Give each of the three releases ρ/3.
3. L2 sensitivities under replace-one (one row removed, another added: twice the one-row change):
   - λ_min(XᵀX): 2·B² (one x·xᵀ moves every eigenvalue by at most ‖x‖², Weyl);
   - upper triangle of XᵀX: 2·B² (‖x·xᵀ‖_F = ‖x‖²);
   - Xᵀy: 2·B·1.
   (Add/remove alternative: half of each.)
4. σ = Δ₂ / √(2·ρ/3) for each release. Noise only the upper triangle and mirror it.
5. Ridge (post-processing): λ = max(0, 2·σ_A·√(D+1) − noisy λ_min) plus a small floor; 2σ√d is roughly the spectral norm of the symmetric noise matrix.
6. `beta <- numpy.linalg.solve(A_noisy + λ·I, v_noisy)`; predictions on public inputs use the same centring and scaling, then undo the label scaling.

## Privacy accounting
Replace-one sensitivities. Total ρ = sum of the three Δ₂²/(2σ²) = ρ; convert back to ε at the given δ and assert it is ≤ ε. Everything after the three noisy releases is post-processing.

## Pitfalls
- Bounds from the data's min/max (the task's bounds must be public by design).
- Centring features or label on the data mean: the shift is then a private statistic that leaks and breaks the sensitivity analysis. Use the public midpoints. Skipping centring is valid but much noisier, because sensitivities scale with the largest absolute value instead of the half-width.
- Clamping the prediction inputs: shift and scale them, but do not clamp (they are public).
- No ridge: a noisy XᵀX can be indefinite and the solve explodes.
- At small N (hundreds of rows) and ε ≈ 1, any DP regression is noisy; fewer features and tighter public bounds help more than tuning.
