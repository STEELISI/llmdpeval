---
id: numpy-ols-noisy-moments-se
title: OLS with several regressors and one-hot dummies from one noisy moment vector, with noise-aware standard errors
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: regression
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.triu_indices
  - numpy.linalg.solve
  - numpy.linalg.inv
  - math.erfc
source_url: https://doi.org/10.29012/jpc.660
source_license: CC-BY-NC-ND-4.0 (paper; ideas paraphrased)
---

## Goal
Fit y ~ intercept + numeric columns + dummy columns of categorical variables and report each coefficient with a standard error and p-value that include the DP noise, so significance claims stay honest. Replace-one neighbours (bounded DP, n public). Sufficient statistics perturbation; sampling plus privacy noise in the standard error as in Wang, Kifer & Lee (JPC 2019).

## Steps
1. Public inputs only: bounds `(L, U)` per numeric column and for y (codebook, physical limits), and the full level list of each categorical column (codebook). Never min/max, mean/std or `unique()` of the data.
2. Build Z: a constant 1; each numeric column clamped and min-maxed to [0, 1] with its public bounds; one 0/1 dummy per non-baseline public level; y min-maxed to [0, 1] last. Public levels mean the columns are the same in every run.
3. Row bound: each block (intercept, numeric column, one-hot group, y) adds at most 1 to sum|z| and to sum z². With R blocks, one row moves the upper triangle of ZᵀZ by at most (R² + R)/2 in L1; replace-one = remove + add, so `SENS <- R² + R`.
4. Release the whole upper triangle at once: `noisy <- triu(ZᵀZ) + laplace(0, SENS/EPS)` per entry. Rebuild the symmetric matrix M.
5. Post-processing: `beta <- solve(M_xx, M_xy)`; `sigma² <- (M_yy − beta·M_xy)/(n − p)`; sampling covariance `sigma² · inv(M_xx)`.
6. Noise variance: redraw fresh Laplace noise of the same public scale on M many times, re-solve, take the variance of the coefficients. SE = sqrt(sampling variance + noise variance); p = erfc(|beta/SE| / √2).
7. Convert slopes back: multiply by (U_y − L_y) / (U_x − L_x) (dummies: by U_y − L_y). p-values do not change.

## Privacy accounting
One Laplace release of a vector with L1 sensitivity SENS under replace-one (n public): `epsilon_spent <- SENS / scale`, assert <= EPS. Solving, the SE simulation and p-values use only the released vector and public numbers, so they are free.

## Pitfalls
- A SE from sampling variance alone: with many dummies the DP noise dominates and a naive test reports false significance.
- One-hot columns from `get_dummies` on each split: the columns differ between runs; use the public level list.
- Standardising with the data mean/std, or bounds read from the data.
- Releasing each entry of ZᵀZ with its own epsilon share instead of one vector release.
- With many columns M can be badly conditioned; a non-significant result is then the honest answer.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
