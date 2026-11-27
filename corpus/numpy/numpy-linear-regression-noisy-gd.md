---
id: numpy-linear-regression-noisy-gd
title: Linear regression by clipped noisy gradient descent (DP-GD, zCDP accounting) in NumPy
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
  - numpy.linalg.norm
  - numpy.minimum
source_url: https://arxiv.org/abs/1607.00133
source_license: original code (algorithm from cited paper)
---

## Goal
Fit a multi-feature linear model with (ε, δ)-DP by full-batch gradient descent where each record's gradient is clipped to norm C and Gaussian noise is added to the summed gradient every step. The clip-and-noise step is from DP-SGD (Abadi et al., CCS 2016); here without subsampling, accounted with zCDP (Bun & Steinke 2016). Neighbouring relation: replace-one (n public).

## Steps
1. Public bounds; centre features and label on their range midpoints and scale to [−1, 1]; clamp; add an intercept column. N is public (replace-one relation).
2. Budget: √ρ = √(ln(1/δ) + ε) − √(ln(1/δ)). T steps (public).
3. Sensitivity of the summed clipped gradient under replace-one: 2C (one clipped gradient swapped for another). σ = 2C·√(T / (2ρ)).
4. Each step:
   - per-record gradient g_i = (x_iᵀw − y_i)·x_i;
   - clip: g_i ← g_i·min(1, C/‖g_i‖);
   - `w <- w − lr·(Σ g_i + rng.normal(0, σ, size=d)) / N`.
5. Release w (and predictions on public inputs, free).

## Privacy accounting
Each step is a Gaussian mechanism with ρ_step = (2C)²/(2σ²); T steps sum to ρ; convert back with ε = ρ + 2√(ρ·ln(1/δ)) and assert ≤ ε. Every intermediate w may be released too: the bound already covers them.

## Pitfalls
- Clipping the averaged gradient instead of each record's gradient: no sensitivity bound.
- Noise scale per step without the √T factor (or claiming ε per step "because each step is ε").
- Dividing by the exact N under add/remove: N must be public by design for this recipe.
- Many steps at small N: noise grows with √T; prefer few steps and a larger learning rate, or SSP (`numpy-linear-regression-ssp-gaussian`) for few features.
- Subsampling amplification needs a proper RDP accountant; this recipe does not claim it.
