---
id: numpy-linear-regression-ssp-centred-public-n
title: Multi-feature linear regression from four noisy sums, centred on public midpoints, N public (Laplace) in NumPy
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
  - numpy.triu_indices
  - numpy.outer
  - numpy.eye
  - numpy.linalg.solve
source_url: https://arxiv.org/abs/1803.02596
source_license: original code (algorithm from cited paper)
---

## Goal
Fit least squares on D features with pure ε-DP by sufficient statistics perturbation (SSP; Vu & Slavkovic 2009, Wang 2018): release four noisy sums (Σx, Σy, upper triangle of XᵀX, Xᵀy) and solve from them. Relation: replace one record; N is public. Laplace mechanism (Dwork & Roth 2014, Thm. 3.6), sequential composition (Cor. 3.15).

## Steps
1. Public bounds [L_j, U_j] per feature and [L_y, U_y] for the label, fixed before seeing the data. Public midpoints c_j = (L_j + U_j)/2 and half-widths h_j = (U_j − L_j)/2.
2. Clip to the bounds, THEN subtract the public midpoints: every centred value lies in [−h, h].
3. Four statistics, with replace-one L1 sensitivities (a value moves anywhere inside its range):
   - S_x = Σ x (D entries): Σ_j 2h_j;
   - S_y = Σ y: 2h_y;
   - upper triangle of XᵀX: diagonal x_j² ∈ [0, h_j²] moves by h_j²; off-diagonal x_j·x_k moves by 2h_j·h_k;
   - Xᵀy (D entries): Σ_j 2h_j·h_y.
4. Split ε over the four sums (shares stated in code, summing to 1); scale_i = Δ_i / ε_i; add `rng.laplace` noise of that scale to every entry of sum i.
5. Post-processing with the public N: means m_x = S_x/N, m_y = S_y/N; covariance C = XᵀX/N − m_x·m_xᵀ; cross term c = Xᵀy/N − m_x·m_y; slope β = solve(C + λI, c) with a small public ridge λ; intercept = m_y − m_xᵀβ.
6. Predict at public inputs: subtract the same midpoints (no clipping), apply β and the intercept, add back c_y.

## Privacy accounting
ε_spent = Σ Δ_i / scale_i over the four releases; assert it equals the budget. Everything after step 4 is post-processing.

## Pitfalls
- Not centring: on raw values a squared feature moves by up to U² − L² (e.g. [1000, 5000]: 24,000,000) instead of h² (4,000,000), and the sums Σx and Σy then carry large values whose noise is amplified again when C = XᵀX/N − m_x·m_xᵀ subtracts two big, noisy numbers. The further the range sits from 0, the worse it gets; at small N this is often the difference between a usable and a useless fit.
- Centring on the data mean: the mean is a statistic of the private data, so the shift leaks it and the sensitivity analysis (which assumes a fixed shift) no longer holds. Use the public midpoint.
- Bounds equal to the data's min/max: the bounds are then private.
- Treating N as public under add/remove: it is not; use the add/remove recipe (`numpy-linear-regression-sufficient-stats`).
- Fixed seeds such as `default_rng(42)`: the noise becomes predictable. Call `default_rng()` with no argument.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
