---
id: diffprivlib-ols-noisy-moments-se
title: OLS with several regressors and one-hot dummies from one noisy moment vector, with noise-aware standard errors
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: regression
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.mechanisms.Laplace.randomise
  - diffprivlib.mechanisms.Laplace.variance
  - diffprivlib.accountant.BudgetAccountant
  - numpy.linalg.solve
  - math.erfc
source_url: https://doi.org/10.29012/jpc.660
source_license: CC-BY-NC-ND-4.0 (paper; ideas paraphrased)
---

## Goal
Fit y ~ intercept + numeric columns + dummy columns of categorical variables and report each coefficient with a standard error and p-value that include the DP noise, so significance claims stay honest. Replace-one neighbours (bounded DP, n public). diffprivlib's LinearRegression gives coefficients but no standard errors, so release the moment vector with diffprivlib's Laplace mechanism and solve as post-processing (sufficient statistics perturbation; noise-aware intervals as in Wang, Kifer & Lee, JPC 2019).

## Steps
1. Public inputs only: bounds `(L, U)` per numeric column and for y (codebook, physical limits), and the full level list of each categorical column (codebook). Never min/max, mean/std or `unique()` of the data.
2. Build Z: a constant 1; each numeric column clamped and min-maxed to [0, 1] with its public bounds; one 0/1 dummy per non-baseline public level; y min-maxed to [0, 1] last. Public levels mean the columns are the same in every run.
3. Row bound: each block (intercept, numeric column, one-hot group, y) adds at most 1 to sum|z| and to sum z². With R blocks, one row moves the upper triangle of ZᵀZ by at most (R² + R)/2 in L1; replace-one = remove + add, so `SENS <- R² + R`.
4. `mech <- Laplace(epsilon=EPS, sensitivity=SENS)`; `noisy <- [mech.randomise(v) for v in triu(ZᵀZ)]` (an L1 vector release is this noise on every entry); `acc.spend(EPS, 0)` once. Noise scale for step 6: `sqrt(mech.variance(0) / 2)`. Rebuild the symmetric matrix M.
5. Post-processing: `beta <- solve(M_xx, M_xy)`; `sigma² <- (M_yy − beta·M_xy)/(n − p)`; sampling covariance `sigma² · inv(M_xx)`.
6. Noise variance: redraw fresh Laplace noise of the same public scale on M many times, re-solve, take the variance of the coefficients. SE = sqrt(sampling variance + noise variance); p = erfc(|beta/SE| / √2).
7. Convert slopes back: multiply by (U_y − L_y) / (U_x − L_x) (dummies: by U_y − L_y). p-values do not change.

## Privacy accounting
One Laplace release of a vector with L1 sensitivity SENS under replace-one (n public), recorded once in the accountant: `epsilon_spent <- acc.total()[0]`, assert <= EPS. Solving, the SE simulation and p-values use only the released vector and public numbers, so they are free.

## Pitfalls
- A SE from sampling variance alone: with many dummies the DP noise dominates and a naive test reports false significance.
- One-hot columns from `get_dummies` on each split: the columns differ between runs; use the public level list.
- Standardising with the data mean/std, or bounds read from the data.
- `models.LinearRegression` alone: it has no standard errors, so no honest test.
- Releasing each entry of ZᵀZ with its own epsilon share instead of one vector release.
- With many columns M can be badly conditioned; a non-significant result is then the honest answer.
