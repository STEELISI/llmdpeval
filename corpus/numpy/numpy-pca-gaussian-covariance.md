---
id: numpy-pca-gaussian-covariance
title: PCA from a Gaussian-noised second-moment matrix (Analyze Gauss) in NumPy
framework: numpy
framework_version: "any"
mechanism: gaussian
task_type: pca
dp_notion: approx
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.normal
  - numpy.clip
  - numpy.triu_indices
  - numpy.triu
  - numpy.linalg.eigh
source_url: https://doi.org/10.1145/2591796.2591883
source_license: original code (algorithm from cited paper)
---

## Goal
Release the top principal directions of several numeric columns with (ε, δ)-DP by adding symmetric Gaussian noise to XᵀX once and taking its eigenvectors. "Analyze Gauss", Dwork, Talwar, Thakurta & Zhang, STOC 2014 (doi 10.1145/2591796.2591883); zCDP accounting (Bun & Steinke 2016). Neighbouring relation: replace-one (n public).

## Steps
1. Public bounds per column. Centre each column on its range midpoint, scale by its half-width to [−1, 1], clamp, then divide each row by √D so ‖x‖ ≤ 1.
2. A = XᵀX (D×D). One row changes A by x·xᵀ, whose Frobenius norm is ‖x‖² ≤ 1. Replace-one removes one such term and adds another: L2 sensitivity 2 (safe bound; add/remove alternative: 1).
3. ρ from (ε, δ): √ρ = √(ln(1/δ) + ε) − √(ln(1/δ)); σ = Δ/√(2ρ).
4. Noise the upper triangle (diagonal included) with N(0, σ²) and mirror it, so the noisy matrix stays symmetric: the upper-triangle vector of x·xᵀ also has L2 norm ≤ 1.
5. `vals, vecs <- numpy.linalg.eigh(A_noisy)`; take the columns for the k largest eigenvalues. Post-processing.
6. Optional: report the share of (noisy) variance explained, also post-processing.

## Privacy accounting
One Gaussian release with ρ = Δ²/(2σ²), Δ = 2 (replace-one); ε_spent = ρ + 2√(ρ ln(1/δ)), asserted ≤ ε. Projecting public or new data on the released directions is free; projecting the private data and publishing the scores is NOT free (it is a new query on the data).

## Pitfalls
- Centring on the data mean: not public. Either centre on public midpoints (as here) or spend budget on a DP mean first.
- Row norm above 1: the sensitivity is then max‖x‖², not 1.
- Noising every entry of the full matrix independently: the result is not symmetric and the sensitivity accounting double-counts off-diagonal entries.
- Standardising with the data's standard deviation.
